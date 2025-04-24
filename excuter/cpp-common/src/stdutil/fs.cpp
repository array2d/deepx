#include "fs.hpp"
#include <fstream>
 
namespace stdutil
{
    string filename(const string &path)
    {
        return path.substr(path.find_last_of('/') + 1);
    }

    /*
    std::ios::binary 二进制打开
    std::ios::in 读
    std::ios::out 写，如果文件不存在，则创建文件
    std::ios::trunc 如果文件存在，则清空文件
    */

    void save(const byte *data, size_t size, const string &path)
    {

        ofstream ofs(path, ios::binary | ios::out | ios::trunc);
        ofs.write(reinterpret_cast<const char *>(data), size);
        ofs.close();
    }

    void load(const string &path,byte *data,size_t target_size){
        ifstream ifs(path, ios::binary | ios::in);
        if (!ifs.is_open())
        {
            throw std::runtime_error("Failed to open file: " + path);
        }

        ifs.seekg(0, ios::end);
        size_t size = ifs.tellg();
        ifs.seekg(0, ios::beg);
        if(size!=target_size){
            throw std::runtime_error("file size mismatch: " + path);
        }
        ifs.read(reinterpret_cast<char *>(data), size);
        if (ifs.fail())
        {
            throw std::runtime_error("Failed to read file: " + path);
        }
        ifs.close();
    }

    std::pair<size_t,shared_ptr<byte[]>> load(const string &path)
    {
        ifstream ifs(path, ios::binary | ios::in);
        if (!ifs.is_open())
        {
            throw std::runtime_error("Failed to open file: " + path);
        }
        ifs.seekg(0, ios::end);
        size_t size = ifs.tellg();
        ifs.seekg(0, ios::beg);
        shared_ptr<byte[]> data(new byte[size]);
        ifs.read(reinterpret_cast<char *>(data.get()), size);
        if (ifs.fail())
        {
            throw std::runtime_error("Failed to read file: " + path);
        }
        ifs.close();
        return std::make_pair(size, data);
    }
}