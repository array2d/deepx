#include "string.hpp"

namespace stdutil
{
    void trimspace(string &str)
    {
        str.erase(0, str.find_first_not_of(" "));
        str.erase(str.find_last_not_of(" ") + 1);
    }

    void trim(string &str,const string &chars)
    {
        str.erase(0, str.find_first_not_of(chars));
        str.erase(str.find_last_not_of(chars) + 1);
    }
} // namespace stdutil