#ifndef __CLIENT_UNIXSOCKETSERVER_HPP__
#define __CLIENT_UNIXSOCKETSERVER_HPP__

#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <unistd.h>
#include <functional>

namespace client
{
    class unixsocketserver
    {
    private:
        std::string socket_path;
        int sockfd;
        struct sockaddr_un servaddr, cliaddr; // 修改为使用完整类型
        char* buffer;        // 改为指针类型
        const int buffer_size; // 新增缓冲区大小成员
        socklen_t len;
        ssize_t n;

    public:
        unixsocketserver(const std::string &path, const int buffersize);
        ~unixsocketserver();
        void start();
        using handlefunc = std::function<void(char *buffer)>;
        handlefunc func;
    };
}

#endif