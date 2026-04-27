#pragma once

#include <string>

namespace deepx::mps
{
struct DeviceInfo
{
    std::string name;
    bool supports_mps{false};
};

DeviceInfo get_default_device_info();
}
