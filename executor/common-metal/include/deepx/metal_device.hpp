#pragma once

#include <string>

namespace deepx::metal
{
struct DeviceInfo
{
    std::string name;
    bool supports_metal{false};
};

DeviceInfo get_default_device_info();
}
