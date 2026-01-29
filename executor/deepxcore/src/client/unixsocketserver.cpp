#include "client/unixsocketserver.hpp"

namespace client
{
    unixsocketserver::unixsocketserver(const std::string& path, const int buffersize)
        : socket_path(path), buffer_size(buffersize)
    {
        buffer = new char[buffer_size];
        if (!buffer) {
            throw std::bad_alloc();
        }
    }
    unixsocketserver::~unixsocketserver()
    {
        delete[] buffer;
        if (sockfd > 0)
        {
            close(sockfd);
            unlink(socket_path.c_str());
        }
    }
    void unixsocketserver::start()
    {
         // 创建Unix domain socket
        if ((sockfd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
            perror("socket creation failed");
            exit(EXIT_FAILURE);
        }

        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sun_family = AF_UNIX;
        strncpy(servaddr.sun_path, socket_path.c_str(), sizeof(servaddr.sun_path)-1);

        // 删除可能存在的旧socket文件
        unlink(socket_path.c_str());

        if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
            perror("bind failed");
            close(sockfd);
            exit(EXIT_FAILURE);
        }

        if (listen(sockfd, 5) < 0) {
            perror("listen failed");
            close(sockfd);
            exit(EXIT_FAILURE);
        }

        while (true) {
            len = sizeof(cliaddr);
            int client_fd = accept(sockfd, (struct sockaddr *)&cliaddr, &len);
            if (client_fd < 0) {
                perror("accept failed");
                continue;
            }

            n = read(client_fd, buffer, buffer_size - 1);
            if (n > 0) {
                buffer[n] = '\0';
                std::cout << "Received message: " << buffer << std::endl;
                func(buffer);
            }
            close(client_fd);
        }
        close(sockfd);
    }
}