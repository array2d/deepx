#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "deepx/metal_device.hpp"

namespace deepx::metal
{
DeviceInfo get_default_device_info()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    DeviceInfo info;

    if (!device)
    {
        info.name = "none";
        info.supports_metal = false;
        return info;
    }

    info.name = std::string([[device name] UTF8String]);
    info.supports_metal = true;
    return info;
}
}
