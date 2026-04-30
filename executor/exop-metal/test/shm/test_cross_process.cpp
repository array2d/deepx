// 集成验证: heap-metal 创建 tensor → op-metal 通过 shm 访问并 GPU 计算
//
// 用法:
//   模拟 heap 先创建 tensor 并写入数据:
//     ./test_cross_process create <shm_name> <count>
//   模拟 op-metal 访问 tensor 并 GPU 计算:
//     ./test_cross_process compute <shm_name> <count>
//
// 手动测试流程:
//   1. ./test_cross_process create /deepx_t_test 1024
//   2. ./test_cross_process compute /deepx_t_test 1024

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

static size_t page_size() {
    static long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? (size_t)ps : 16384;
}
static size_t page_align(size_t n) {
    size_t ps = page_size();
    return (n + ps - 1) & ~(ps - 1);
}

// ── heap 角色：创建 shm tensor ──────────────────────────────────
static int create_tensor(const char *shm_name, int count) {
    size_t single  = count * sizeof(float);
    size_t off_a   = 0;
    size_t off_b   = page_align(single);
    size_t off_c   = off_b + page_align(single);
    size_t total   = off_c + page_align(single);

    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
    if (fd < 0) { perror("shm_open"); return 1; }
    if (ftruncate(fd, total) < 0) { perror("ftruncate"); return 1; }

    uint8_t *base = (uint8_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                     MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);

    float *A = (float *)(base + off_a);
    float *B = (float *)(base + off_b);

    for (int i = 0; i < count; i++) {
        A[i] = (float)(i + 1);
        B[i] = (float)(count - i);
    }
    printf("[heap] tensor created: shm=%s count=%d A[0]=%f B[0]=%f\n",
           shm_name, count, A[0], B[0]);
    munmap(base, total);
    return 0;
}

// ── op-metal 角色：打开 shm，GPU 计算 ──────────────────────────────
static int compute_tensor(const char *shm_name, int count) {
    @autoreleasepool {
        size_t single  = count * sizeof(float);
        size_t off_a   = 0;
        size_t off_b   = page_align(single);
        size_t off_c   = off_b + page_align(single);
        size_t total   = off_c + page_align(single);

        int fd = shm_open(shm_name, O_RDWR, 0600);
        if (fd < 0) { perror("exop-metal shm_open"); return 1; }

        uint8_t *base = (uint8_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                         MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { perror("exop-metal mmap"); return 1; }
        close(fd);

        float *A = (float *)(base + off_a);
        float *B = (float *)(base + off_b);
        float *C = (float *)(base + off_c);

        // Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { printf("[exexop-metal] FAIL: no Metal device\n"); return 1; }
        printf("[exexop-metal] device: %s\n", [[device name] UTF8String]);

        // Compile kernel
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
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                  options:[MTLCompileOptions new]
                                                    error:&err];
        if (!lib) {
            printf("[exexop-metal] FAIL: compile: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"add_f32"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // ★ 关键：从 shm 指针创建 MTLBuffer (no-copy)
        id<MTLBuffer> bufA = [device newBufferWithBytesNoCopy:A length:single
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        id<MTLBuffer> bufB = [device newBufferWithBytesNoCopy:B length:single
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        id<MTLBuffer> bufC = [device newBufferWithBytesNoCopy:C length:single
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        uint32_t n = (uint32_t)count;
        id<MTLBuffer> bufN = [device newBufferWithBytes:&n length:sizeof(n)
                                                options:MTLResourceStorageModeShared];

        if (!bufA || !bufB || !bufC) {
            printf("[exexop-metal] FAIL: newBufferWithBytesNoCopy returned nil\n");
            return 1;
        }
        printf("[exexop-metal] MTLBuffers from shm OK\n");

        // Dispatch
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBuffer:bufN offset:0 atIndex:3];
        NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            printf("[exexop-metal] FAIL: GPU error: %s\n",
                   [[cmd.error localizedDescription] UTF8String]);
            return 1;
        }
        printf("[exexop-metal] GPU kernel done.\n");

        // Verify
        int errors = 0;
        for (int i = 0; i < count; i++) {
            float expected = (float)(i + 1) + (float)(count - i);
            if (fabsf(C[i] - expected) > 1e-6f) {
                if (errors < 5)
                    printf("  MISMATCH [%d]: got=%f expected=%f\n", i, C[i], expected);
                errors++;
            }
        }

        munmap(base, total);

        if (errors == 0) {
            printf("[exexop-metal] PASS: all %d elements correct.\n", count);
            return 0;
        } else {
            printf("[exexop-metal] FAIL: %d / %d mismatches.\n", errors, count);
            return 1;
        }
    }
}

// ── main ────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s create <shm_name> <count>\n", argv[0]);
        fprintf(stderr, "       %s compute <shm_name> <count>\n", argv[0]);
        return 1;
    }
    const char *mode     = argv[1];
    const char *shm_name = argv[2];
    int count = (argc > 3) ? atoi(argv[3]) : 1024;

    if (strcmp(mode, "create") == 0) {
        return create_tensor(shm_name, count);
    } else if (strcmp(mode, "compute") == 0) {
        return compute_tensor(shm_name, count);
    }
    fprintf(stderr, "Unknown mode: %s\n", mode);
    return 1;
}
