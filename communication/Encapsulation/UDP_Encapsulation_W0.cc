/*
这个是将UDP的API进行封装成为Recv和Send的函数
预期直接提供给Pytorch进行调用

该文件是完成双机（Worker0）测试
能够正常接收

需要注意得先启动Recv再启动Send
而且若Recv未收到会出现阻塞
需要超时退出？
*/
#include <iostream>
#include <unordered_map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <thread>
#include <unistd.h>
#include <cstring>



class UDPCommunicator {
private:
    int sockfd;
    std::unordered_map<int, sockaddr_in> peer_addr_map;  // Maps rank to sockaddr_in

public:
    UDPCommunicator(const std::unordered_map<int, std::pair<std::string, int>>& addr_map, int my_rank) {
        // 创建socket
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) {
            perror("Socket creation failed");
            exit(EXIT_FAILURE);
        }

        // 绑定本地地址
        sockaddr_in my_addr;
        auto my_address = addr_map.at(my_rank);
        my_addr.sin_family = AF_INET;
        my_addr.sin_port = htons(my_address.second);
        my_addr.sin_addr.s_addr = inet_addr(my_address.first.c_str());
        if (bind(sockfd, (struct sockaddr *)&my_addr, sizeof(my_addr)) < 0) {
            perror("Bind failed");
            exit(EXIT_FAILURE);
        }

        // 初始化peer地址信息
        for (const auto& addr : addr_map) {
            sockaddr_in peer_addr;
            peer_addr.sin_family = AF_INET;
            peer_addr.sin_port = htons(addr.second.second);
            peer_addr.sin_addr.s_addr = inet_addr(addr.second.first.c_str());
            peer_addr_map[addr.first] = peer_addr;
        }
    }

    ~UDPCommunicator() {
        close(sockfd);
    }

    void send(const void* buf, size_t len, int dst_rank) {
        const auto& dest_addr = peer_addr_map[dst_rank];
        ssize_t sent = sendto(sockfd, buf, len, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (sent < 0) {
            perror("Send failed");
        }
    }

    void recv(void* buf, size_t len, int src_rank) {
        sockaddr_in src_addr;
        socklen_t addrlen = sizeof(src_addr);
        ssize_t received = recvfrom(sockfd, buf, len, 0, (struct sockaddr *)&src_addr, &addrlen);
        if (received < 0) {
            perror("Receive failed");
        }
    }
};

int main() {
    std::unordered_map<int, std::pair<std::string, int>> addr_map = {
        {0, {"192.168.11.150", 8080}},
        {1, {"192.168.11.226", 8081}}
    };

    int my_rank = 0; // This node's rank
    UDPCommunicator comm0(addr_map, my_rank);
    
    char msg[] = "Hello, world!";
    comm0.send(msg, sizeof(msg), 1); // Send to rank 1

    return 0;
}

