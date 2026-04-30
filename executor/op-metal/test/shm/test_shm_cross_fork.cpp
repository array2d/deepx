// Stage 2: fork 跨进程验证 — parent GPU compute, child CPU verify via POSIX shm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cerrno>

static const char *SHM_NAME  = "/deepx_shm_stage2";
static const char *SEM_PARENT = "/deepx_sem_parent_done";
static const int   COUNT     = 2048;

static size_t page_size() {
    static size_t sz = 0;
    if (!sz) sz = sysconf(_SC_PAGESIZE);
    return sz;
}
static size_t page_align(size_t n) {
    size_t ps = page_size();
    return (n + ps - 1) & ~(ps - 1);
}

// ── child 进程：等 parent GPU 完成后，从 shm 读 C 验证 ──────────
static int child_main(const char *shm_name, const char *sem_name) {
    // 等 parent
    sem_t *sem = sem_open(sem_name, 0);
    if (sem == SEM_FAILED) {
        perror("child sem_open");
        // 可能 parent 还没创建，重试
        for (int retry = 0; retry < 100; retry++) {
            usleep(10000);
            sem = sem_open(sem_name, 0);
            if (sem != SEM_FAILED) break;
        }
        if (sem == SEM_FAILED) { printf("CHILD FAIL: cannot open semaphore\n"); return 1; }
    }
    printf("[child] waiting for parent GPU...\n");
    sem_wait(sem);
    sem_close(sem);
    printf("[child] semaphore acquired, reading shm...\n");

    int fd = shm_open(shm_name, O_RDONLY, 0600);
    if (fd < 0) { perror("child shm_open"); return 1; }

    size_t single_bytes = COUNT * sizeof(float);
    size_t off_a = 0;
    size_t off_b = page_align(single_bytes);
    size_t off_c = off_b + page_align(single_bytes);
    size_t total  = off_c + page_align(single_bytes);

    uint8_t *base = (uint8_t *)mmap(NULL, total, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("child mmap"); return 1; }
    close(fd);

    float *A = (float *)(base + off_a);
    float *B = (float *)(base + off_b);
    float *C = (float *)(base + off_c);

    int errors = 0;
    for (int i = 0; i < COUNT; i++) {
        float expected = (float)(i + 1) + (float)(COUNT - i);
        if (fabsf(C[i] - expected) > 1e-6f) {
            if (errors < 5) {
                printf("  [child] MISMATCH [%d]: A=%f B=%f got=%f expected=%f\n",
                       i, A[i], B[i], C[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("[child] PASS: all %d elements correct.\n", COUNT);
    } else {
        printf("[child] FAIL: %d / %d mismatches.\n", errors, COUNT);
    }
    munmap(base, total);
    return (errors == 0) ? 0 : 1;
}

// ── parent 进程：GPU compute ──────────────────────────────────────
static int parent_main(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> pso,
                       const char *shm_name, const char *sem_name) {
    @autoreleasepool {
        // 创建/打开 shm
        shm_unlink(shm_name); // 确保干净
        int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
        if (fd < 0) { perror("parent shm_open"); return 1; }

        size_t single_bytes = COUNT * sizeof(float);
        size_t off_a = 0;
        size_t off_b = page_align(single_bytes);
        size_t off_c = off_b + page_align(single_bytes);
        size_t total  = off_c + page_align(single_bytes);

        if (ftruncate(fd, total) < 0) { perror("ftruncate"); return 1; }

        uint8_t *base = (uint8_t *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { perror("mmap"); return 1; }
        close(fd);

        float *A = (float *)(base + off_a);
        float *B = (float *)(base + off_b);
        float *C = (float *)(base + off_c);

        // CPU 填充
        for (int i = 0; i < COUNT; i++) {
            A[i] = (float)(i + 1);
            B[i] = (float)(COUNT - i);
        }

        // 从 shm 创建 MTLBuffer
        id<MTLBuffer> bufA = [device newBufferWithBytesNoCopy:A length:single_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        id<MTLBuffer> bufB = [device newBufferWithBytesNoCopy:B length:single_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        id<MTLBuffer> bufC = [device newBufferWithBytesNoCopy:C length:single_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        uint32_t n = COUNT;
        id<MTLBuffer> bufN = [device newBufferWithBytes:&n length:sizeof(n)
                                                options:MTLResourceStorageModeShared];

        if (!bufA || !bufB || !bufC) {
            printf("[parent] FAIL: MTLBuffer no-copy returned nil\n");
            return 1;
        }
        printf("[parent] MTLBuffers from shm OK\n");

        // GPU dispatch
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBuffer:bufN offset:0 atIndex:3];

        NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
        [enc dispatchThreads:MTLSizeMake(COUNT, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            printf("[parent] FAIL: GPU error: %s\n",
                   [[cmd.error localizedDescription] UTF8String]);
            return 1;
        }
        printf("[parent] GPU kernel done, signaling child...\n");

        // 创建 semaphore 并 post
        sem_t *sem = sem_open(sem_name, O_CREAT | O_EXCL, 0600, 0);
        if (sem == SEM_FAILED) {
            perror("sem_open");
            return 1;
        }
        sem_post(sem);
        sem_close(sem);

        // 等 child 读完
        munmap(base, total);
    }
    return 0;
}

// ── entry ─────────────────────────────────────────────────────────
int main() {
    @autoreleasepool {
        // 公共 Metal 初始化
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { printf("FAIL: no Metal device\n"); return 1; }
        printf("Device: %s\n", [[device name] UTF8String]);

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
            printf("FAIL: compile: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"add_f32"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn
                                                                                error:&err];
        if (!pso) {
            printf("FAIL: pipeline: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // 清理 semaphore
        sem_unlink(SEM_PARENT);

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            // child — 注意：fork 后不能直接用 ObjC 对象 (MTLDevice 等)
            // child 只做 shm + CPU 读取，不碰 Metal
            _exit(child_main(SHM_NAME, SEM_PARENT));
        } else {
            int parent_ret = parent_main(device, queue, pso, SHM_NAME, SEM_PARENT);
            int status;
            waitpid(pid, &status, 0);
            int child_ret = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

            // 清理
            shm_unlink(SHM_NAME);
            sem_unlink(SEM_PARENT);

            if (parent_ret != 0 || child_ret != 0) {
                printf("OVERALL: FAIL (parent=%d child=%d)\n", parent_ret, child_ret);
                return 1;
            }
            printf("OVERALL: PASS — cross-process shm + GPU verified.\n");
            return 0;
        }
    }
}
