#ifndef DEEPX_TENSORFUNC_AUTHORS_HPP
#define DEEPX_TENSORFUNC_AUTHORS_HPP

#include "string"

namespace deepx::tensorfunc{
    using namespace std;
    class default_{
    public:
        static std::string name() { return "default"; }
    };
    
    class miaobyte{
    public:
        static std::string name() { return "miaobyte"; }
    };
    
    class cblas{
    public:
        static std::string name() { return "cblas"; }
    };

    class cublas{
    public:
        static std::string name() { return "cublas"; }
    };
}

#endif
