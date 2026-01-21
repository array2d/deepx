#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "deepx/mps_device.hpp"

namespace deepx::mps
{
DeviceInfo get_default_device_info()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    DeviceInfo info;

    if (!device)
    {
        info.name = "none";
        info.supports_mps = false;
        return info;
    }

    info.name = std::string([[device name] UTF8String]);
    info.supports_mps = true;
    return info;
}
}
