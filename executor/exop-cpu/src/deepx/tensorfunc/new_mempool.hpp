#ifndef DEEPX_TENSORFUNC_NEW_MEMPOOL_HPP
#define DEEPX_TENSORFUNC_NEW_MEMPOOL_HPP

#include <jemalloc/jemalloc.h>

namespace deepx::tensorfunc
{
    class MemoryPool
    {
    public:
    static void* Malloc(size_t size) {
        return mallocx(size, MALLOCX_ALIGN(64));  // 64字节对齐，适合SIMD
    }   

    static void Free(void* ptr) {
        dallocx(ptr, 0);
    }
    
    // Realloc: 重新分配内存并保留原数据,主要用于tensor形状改变时的内存重分配
    // 如果新的size小于原size,数据会被截断
    // 如果新的size大于原size,新分配的内存部分不会初始化
    // 如果ptr为nullptr,等同于Malloc
    // 如果size为0,等同于Free
    // 返回新分配的内存指针,如果分配失败返回nullptr

    static void* Realloc(void* ptr, size_t size) {
        return rallocx(ptr, size, MALLOCX_ALIGN(64));
    }

        
    // GetAllocatedSize: 获取已分配内存的实际大小
    // 由于内存对齐,实际分配的内存可能大于请求的size
    // 主要用于内存使用统计和调试
    // 如果ptr为nullptr,返回0
    // 重新分配内存，保留原数据
    static size_t GetAllocatedSize(void* ptr) {
        return sallocx(ptr, 0);
    }
};

}  // namespace deepx::tensorfunc
#endif  // DEEPX_TENSORFUNC_NEW_MEMPOOL_HPP
