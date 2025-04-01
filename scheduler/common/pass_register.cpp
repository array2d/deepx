#include "pass_register.hpp"

namespace deepx
{

    PassRegistry &PassRegistry::instance()
    {
        static PassRegistry registry_ins;
        return registry_ins;
    }

    void PassRegistry::register_pass(const std::string &name, pass_func func)
    {
        registry_[name] = func;
    }
}