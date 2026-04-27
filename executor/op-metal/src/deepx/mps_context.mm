#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "deepx/mps_context.hpp"

namespace deepx::mps
{
MpsContext::MpsContext()
{
    device_ = MTLCreateSystemDefaultDevice();
    if (device_)
    {
        command_queue_ = [device_ newCommandQueue];
    }
}

bool MpsContext::is_valid() const
{
    return device_ != nil;
}

std::string MpsContext::device_name() const
{
    if (!device_)
    {
        return "none";
    }
    return std::string([[device_ name] UTF8String]);
}

id<MTLDevice> MpsContext::device() const
{
    return device_;
}

id<MTLCommandQueue> MpsContext::command_queue() const
{
    return command_queue_;
}
}
