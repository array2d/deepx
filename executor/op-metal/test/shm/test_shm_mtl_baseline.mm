// Stage 1: 单进程验证 POSIX shm → MTLBuffer no-copy → GPU kernel 读写
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

static const char *SHM_NAME = "/deepx_shm_stage1";
static const int   COUNT    = 1024;

// ── helpers ──────────────────────────────────────────────────────────
static size_t page_size() {
    static size_t sz = 0;
    if (!sz) sz = sysconf(_SC_PAGESIZE);
    return sz;
}
static size_t page_align(size_t n) {
    size_t ps = page_size();
    return (n + ps - 1) & ~(ps - 1);
}

// ── main ─────────────────────────────────────────────────────────────
int main() {
    @autoreleasepool {
        // ---- 1. POSIX shm 分配 ----
        int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0600);
        if (fd < 0) { perror("shm_open"); return 1; }

        size_t single_bytes = COUNT * sizeof(float);
        size_t alloc_a = page_align(single_bytes);
        size_t alloc_b = page_align(single_bytes);
        size_t alloc_c = page_align(single_bytes);
        size_t total   = alloc_a + alloc_b + alloc_c;

        if (ftruncate(fd, total) < 0) { perror("ftruncate"); return 1; }

        float *base = (float *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { perror("mmap"); return 1; }
        close(fd);

        float *A = base;
        float *B = (float *)((uint8_t *)base + alloc_a);
        float *C = (float *)((uint8_t *)base + alloc_a + alloc_b);

        // ---- 2. CPU 填充 A, B ----
        for (int i = 0; i < COUNT; i++) {
            A[i] = (float)(i + 1);
            B[i] = (float)(COUNT - i);
        }

        // ---- 3. Metal 设备 & kernel ----
        id<MTLDevice>      device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!device || !queue) { printf("FAIL: no Metal device\n"); return 1; }

        printf("Metal device: %s\n", [[device name] UTF8String]);
        printf("page_size: %zu  total shm: %zu KB\n", page_size(), total / 1024);

        // 运行时编译 Metal kernel
        NSString *src = @""
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void add_f32(device const float* A [[buffer(0)]],\n"
        "                    device const float* B [[buffer(1)]],\n"
        "                    device float*       C [[buffer(2)]],\n"
        "                    constant uint&      n [[buffer(3)]],\n"
        "                    uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid < n) { C[gid] = A[gid] + B[gid]; }\n"
        "}\n";

        MTLCompileOptions *opts = [MTLCompileOptions new];
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            printf("FAIL: compile Metal: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"add_f32"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            printf("FAIL: create pipeline: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        // ---- 4. 从 shm 指针创建 MTLBuffer (no-copy) ----
        // 关键调用: newBufferWithBytesNoCopy
        id<MTLBuffer> bufA = [device newBufferWithBytesNoCopy:A
                                                       length:single_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        id<MTLBuffer> bufB = [device newBufferWithBytesNoCopy:B
                                                       length:single_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        id<MTLBuffer> bufC = [device newBufferWithBytesNoCopy:C
                                                       length:single_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        uint32_t n = COUNT;
        id<MTLBuffer> bufN = [device newBufferWithBytes:&n length:sizeof(n)
                                                options:MTLResourceStorageModeShared];

        if (!bufA || !bufB || !bufC) {
            printf("FAIL: newBufferWithBytesNoCopy returned nil\n");
            return 1;
        }
        printf("bufA=0x%lx bufB=0x%lx bufC=0x%lx (shm pointers)\n",
               (uintptr_t)[bufA contents], (uintptr_t)[bufB contents],
               (uintptr_t)[bufC contents]);
        printf("base=0x%lx -> A=0x%lx B=0x%lx C=0x%lx\n",
               (uintptr_t)base, (uintptr_t)A, (uintptr_t)B, (uintptr_t)C);

        // ---- 5. Dispatch GPU kernel ----
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBuffer:bufN offset:0 atIndex:3];

        NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
        MTLSize tg  = MTLSizeMake(w, 1, 1);
        MTLSize grid = MTLSizeMake(COUNT, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            printf("FAIL: GPU error: %s\n", [[cmd.error localizedDescription] UTF8String]);
            return 1;
        }
        printf("GPU kernel completed.\n");

        // ---- 6. CPU 验证 ----
        int errors = 0;
        for (int i = 0; i < COUNT; i++) {
            float expected = (float)(i + 1) + (float)(COUNT - i);
            if (fabsf(C[i] - expected) > 1e-6f) {
                if (errors < 5) {
                    printf("  MISMATCH [%d]: got=%f expected=%f\n", i, C[i], expected);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("PASS: all %d elements correct.\n", COUNT);
        } else {
            printf("FAIL: %d / %d mismatches.\n", errors, COUNT);
        }

        // ---- 7. 清理 ----
        munmap(base, total);
        shm_unlink(SHM_NAME);

        return (errors == 0) ? 0 : 1;
    }
}
