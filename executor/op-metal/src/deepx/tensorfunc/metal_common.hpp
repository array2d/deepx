#ifndef DEEPX_TENSORFUNC_METAL_COMMON_HPP
#define DEEPX_TENSORFUNC_METAL_COMMON_HPP

#if defined(__APPLE__)
  #include <TargetConditionals.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

#if defined(__APPLE__) && TARGET_OS_OSX
  #if defined(__OBJC__)
    #import <Foundation/Foundation.h>
    #import <Metal/Metal.h>
  #endif
#endif

namespace deepx::metal::common
{
#if defined(__APPLE__) && TARGET_OS_OSX && defined(__OBJC__)
    class MetalKernelRuntime final
    {
    public:
        MetalKernelRuntime()
        {
            device_ = MTLCreateSystemDefaultDevice();
            queue_ = device_ ? [device_ newCommandQueue] : nil;
        }

        bool valid() const { return device_ != nil && queue_ != nil; }

        bool dispatch_binary_1d(const char *kernel_fn,
                                const void *a,
                                const void *b,
                                void *c,
                                uint32_t n,
                                size_t elem_bytes)
        {
            if (!valid() || !kernel_fn) return false;

            @autoreleasepool
            {
                NSError *error = nil;
                id<MTLComputePipelineState> pso = pipeline(kernel_fn, &error);
                if (!pso) return false;

                const size_t bytes = static_cast<size_t>(n) * elem_bytes;
                id<MTLBuffer> bufA = [device_ newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufB = [device_ newBufferWithBytes:b length:bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufC = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufN = [device_ newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared];
                if (!bufA || !bufB || !bufC || !bufN) return false;

                id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bufA offset:0 atIndex:0];
                [enc setBuffer:bufB offset:0 atIndex:1];
                [enc setBuffer:bufC offset:0 atIndex:2];
                [enc setBuffer:bufN offset:0 atIndex:3];

                const NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
                const MTLSize threadsPerThreadgroup = MTLSizeMake(w, 1, 1);
                const MTLSize threadsPerGrid = MTLSizeMake(n, 1, 1);
                [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                std::memcpy(c, [bufC contents], bytes);
                return true;
            }
        }

        bool dispatch_unary_1d(const char *kernel_fn,
                               const void *x,
                               void *y,
                               uint32_t n,
                               size_t elem_bytes)
        {
            if (!valid() || !kernel_fn) return false;

            @autoreleasepool
            {
                NSError *error = nil;
                id<MTLComputePipelineState> pso = pipeline(kernel_fn, &error);
                if (!pso) return false;

                const size_t bytes = static_cast<size_t>(n) * elem_bytes;
                id<MTLBuffer> bufX = [device_ newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufY = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufN = [device_ newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared];
                if (!bufX || !bufY || !bufN) return false;

                id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bufX offset:0 atIndex:0];
                [enc setBuffer:bufY offset:0 atIndex:1];
                [enc setBuffer:bufN offset:0 atIndex:2];

                const NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
                const MTLSize threadsPerThreadgroup = MTLSizeMake(w, 1, 1);
                const MTLSize threadsPerGrid = MTLSizeMake(n, 1, 1);
                [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                std::memcpy(y, [bufY contents], bytes);
                return true;
            }
        }

    private:
        // 加载预编译的 default.metallib（CMake 编译 .metal → .metallib，放在可执行文件同目录）
        id<MTLLibrary> library(NSError **error)
        {
            if (library_) return library_;

            NSString *exePath = [[NSProcessInfo processInfo] arguments][0];
            NSString *exeDir  = [exePath stringByDeletingLastPathComponent];
            NSString *path    = [exeDir stringByAppendingPathComponent:@"default.metallib"];
            NSURL    *url     = [NSURL fileURLWithPath:path];
            library_ = [device_ newLibraryWithURL:url error:error];
            return library_;
        }

        id<MTLComputePipelineState> pipeline(const char *kernel_fn, NSError **error)
        {
            NSString *fnName = [NSString stringWithUTF8String:kernel_fn];
            auto it = pipeline_cache_.find(fnName);
            if (it != pipeline_cache_.end()) return it->second;

            id<MTLLibrary> lib = library(error);
            if (!lib) return nil;

            id<MTLFunction> fn = [lib newFunctionWithName:fnName];
            if (!fn) return nil;

            id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:error];
            if (pso) pipeline_cache_.emplace(fnName, pso);
            return pso;
        }

        id<MTLDevice> device_ = nil;
        id<MTLCommandQueue> queue_ = nil;
        id<MTLLibrary> library_ = nil;
        std::unordered_map<NSString *, id<MTLComputePipelineState>> pipeline_cache_;
    };
#endif
} // namespace deepx::metal::common

#endif // DEEPX_TENSORFUNC_METAL_COMMON_HPP
