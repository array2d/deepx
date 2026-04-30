#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "deepx/metal_context.hpp"

namespace deepx::metal
{
MetalContext::MetalContext()
{
    device_ = MTLCreateSystemDefaultDevice();
    if (device_)
    {
        command_queue_ = [device_ newCommandQueue];
    }
}

bool MetalContext::is_valid() const
{
    return device_ != nil;
}

std::string MetalContext::device_name() const
{
    if (!device_)
    {
        return "none";
    }
    return std::string([[device_ name] UTF8String]);
}

id<MTLDevice> MetalContext::device() const
{
    return device_;
}

id<MTLCommandQueue> MetalContext::command_queue() const
{
    return command_queue_;
}
}
