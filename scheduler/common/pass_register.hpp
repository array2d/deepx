#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>

/**
 * 使用REGISTER_PASS 宏用于注册pass
 */
namespace deepx
{
    using pass_func = std::function<void()>;

    class PassRegistry
    {
    public:
        static PassRegistry& instance();

        void register_pass(const std::string &name, pass_func func);

        PassRegistry(const PassRegistry&) = delete;
        PassRegistry& operator=(const PassRegistry&) = delete;

    private:
        PassRegistry() = default;

    private:
        std::unordered_map<std::string, pass_func> registry_;
    };
}


#define REGISTER_PASS(name, func) \
    struct Register##name { \
        Register##name() { \
            PassRegistry::instance().register_pass(#name, func); \
        } \
    }; \
    static Register##name register_##name;