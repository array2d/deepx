#include <iostream>
#include "deepx/mps_context.hpp"
#include "deepx/mps_device.hpp"

int main()
{
    deepx::mps::MpsContext ctx;
    auto info = deepx::mps::get_default_device_info();

    std::cout << "MPS device: " << info.name
              << " (available=" << (info.supports_mps ? "true" : "false") << ")\n";
    std::cout << "Context valid: " << (ctx.is_valid() ? "true" : "false") << "\n";

    return ctx.is_valid() ? 0 : 1;
}
