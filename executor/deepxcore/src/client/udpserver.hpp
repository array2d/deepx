#ifndef __CLIENT_UDPSERVER_HPP__
#define __CLIENT_UDPSERVER_HPP__

#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <unistd.h>
#include <functional>
#include <queue>

#include "deepx/tf/tf.hpp"
namespace client{
    using namespace std;
    class udpserver
    {
    private:
        int port;
        int sockfd;
        struct sockaddr_in servaddr,cliaddr;
        char buffer[1024];
        socklen_t len;
        ssize_t n;
    public:
        udpserver(int port);
        ~udpserver();
        void start(queue<deepx::tf::TF> &tasks);
        using handlefunc = std::function<void(const char *buffer)>;
        handlefunc func;
        void resp(string str);
    };
}

#endif