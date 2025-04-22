#include <sstream>
#include <queue>

#include "client/udpserver.hpp"

namespace client
{
    using namespace std;
    udpserver::udpserver(int port)
    {
        this->port = port;
    };
    udpserver::~udpserver()
    {
        if (sockfd > 0)
        {
            close(sockfd);
        }
    }
    void udpserver::start(queue<deepx::tf::TF> &queue)
    {
        // 创建UDP套接字
        if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        {
            perror("socket creation failed");
            exit(EXIT_FAILURE);
        }

        memset(&servaddr, 0, sizeof(servaddr));
 
        // 绑定IP和端口
        servaddr.sin_family = AF_INET; // IPv4
        servaddr.sin_addr.s_addr = INADDR_ANY;
        servaddr.sin_port = htons(port);

        if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
        {
            perror("bind failed");
            close(sockfd);
            exit(EXIT_FAILURE);
        }
        while (true)
        {
            len = sizeof(cliaddr);
            n = recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr *)&cliaddr, &len);
            buffer[n] = '\0';
            
            // 新增换行拆分逻辑
            stringstream ss(buffer);
            string line;
            while (getline(ss, line)) {
                if (!line.empty()) {
                    deepx::tf::TF tf;
                    tf.metadata.recv_at = chrono::system_clock::now();
                    tf.parse(line);
                    queue.push(tf);
                }
            }
        }
        close(sockfd);
    }
    void udpserver::resp(string str){
         sendto(sockfd, str.c_str(), str.size(), 0,  // 改为sendto
          (const struct sockaddr *)&cliaddr, sizeof(cliaddr));
    }
}