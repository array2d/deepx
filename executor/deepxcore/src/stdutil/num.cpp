#include "num.hpp"
#include <string>
#include <cctype>
 
bool is_positive_integer(const std::string& str) {
    try {
        int s=std::stoi(str);
        if(s<0){
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}
bool is_integer(const std::string& str) {
   try {
        std::stoi(str);
        return true;
    } catch (...) {
        return false;
    }
}
bool is_float(const std::string& str) {
    try {
        std::stof(str);
        return true;
    } catch (...) {
        return false;
    }
}

