#ifndef STDUTIL_STRING_HPP
#define STDUTIL_STRING_HPP

#include <string>

namespace stdutil
{
    using std::string;

    void trimspace(string &str);
    void trim(string &str,const string &chars=" \t\n\r\f\v");

    string escape_markdown(const string &str);
}


#endif
