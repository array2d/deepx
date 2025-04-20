#include "fs.hpp"

namespace  stdutil{
    string filename(const string &path){
        return path.substr(path.find_last_of('/') + 1);
    }
}