#ifndef DEEPX_STDUTIL_FS_HPP
#define DEEPX_STDUTIL_FS_HPP

#include <string>
#include <memory>

namespace stdutil{

    

    using namespace std;
    string filename(const string &path);

    using byte = unsigned char;

    void save(const byte *data,size_t size,const string &path);
    void load(const string &path,byte *data,size_t target_size);
    pair<size_t,shared_ptr<byte[]>> load(const string &path);
}

#endif // DEEPX_STDUTIL_FS_HPP