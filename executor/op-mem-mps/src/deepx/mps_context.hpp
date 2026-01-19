#pragma once

#include <string>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
@protocol MTLDevice;
@protocol MTLCommandQueue;
#endif

namespace deepx::mps
{
class MpsContext
{
public:
    MpsContext();
    bool is_valid() const;
    std::string device_name() const;

#ifdef __OBJC__
    id<MTLDevice> device() const;
    id<MTLCommandQueue> command_queue() const;
#endif

private:
#ifdef __OBJC__
    id<MTLDevice> device_{nil};
    id<MTLCommandQueue> command_queue_{nil};
#else
    void *device_{nullptr};
    void *command_queue_{nullptr};
#endif
};
}
