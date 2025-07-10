#include "string.hpp"

namespace stdutil
{
    void trimspace(string &str)
    {
        str.erase(0, str.find_first_not_of(" "));
        str.erase(str.find_last_not_of(" ") + 1);
    }

    void trim(string &str, const string &chars)
    {
        str.erase(0, str.find_first_not_of(chars));
        str.erase(str.find_last_not_of(chars) + 1);
    }

    string escape_markdown(const string &str)
    {
        std::string result;
        for (char c : str)
        {
            switch (c)
            {
            case '\\':
                result += "\\\\";
                break;
            case '\"':
                result += "\\\"";
                break;
            case '\'':
                result += "\\\'";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\t':
                result += "\\t";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\b':
                result += "\\b";
                break;
            case '\f':
                result += "\\f";
                break;
            default:
                // 普通字符直接添加
                result += c;
            }
        }
        return result;
    }

} // namespace stdutil