#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <netinet/in.h>
#include <unistd.h>
#include <vector>
#include <arpa/inet.h>

#include <chrono>
#include <thread>

namespace py = pybind11;

// 解决了指针传递的问题，python所传递的是整数类型的地址，需要在传入设置里进行一定的修改
void transfer_tensor(uintptr_t data, size_t num_elements)
{
    float *float_data = reinterpret_cast<float *>(data);
    std::cout << "Received tensor elements: " << std::endl;
    for (size_t i = 0; i < num_elements; ++i)
    {
        std::cout << float_data[i] << " ";
    }
    std::cout << std::endl;
}

void _send(float *data, size_t num_elements)
{

    int sock = 0;
    struct sockaddr_in serv_addr;
    const char *server_ip = "192.168.11.225"; // 服务器IP地址，根据实际情况修改
    int port = 8080;                          // 端口号，根据实际情况修改

    // 创建socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        std::cerr << "Socket creation error" << std::endl;
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    // 将IP地址从文本转换为二进制
    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0)
    {
        std::cerr << "Invalid address / Address not supported" << std::endl;
        return;
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    // 连接到服务器
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        std::cerr << "Connection Failed" << std::endl;
        return;
    }

    // 发送数据
    send(sock, data, num_elements * sizeof(float), 0);
    std::cout << "Tensor sent" << std::endl;

    // 关闭socket
    close(sock);
}

std::vector<float> _recv2(size_t num_elements)
{
    int server_fd, client_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // 创建套接字
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 设置套接字选项
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    // 绑定套接字到地址
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // 监听套接字
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // 接受连接
    if ((client_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0)
    {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    std::vector<float> received_data;
    received_data.resize(num_elements);

    char *buffer = new char[num_elements * sizeof(float)];
    ssize_t bytes_received = 0;
    ssize_t total_bytes_received = 0;

    // 循环接收直到我们得到预期数量的字节
    while (total_bytes_received < num_elements * sizeof(float))
    {
        bytes_received = recv(client_socket, buffer + total_bytes_received, num_elements * sizeof(float) - total_bytes_received, 0);
        if (bytes_received < 0)
        {
            std::cerr << "Failed to receive data" << std::endl;
            delete[] buffer;
            return std::vector<float>();
        }
        total_bytes_received += bytes_received;
    }

    // 将接收到的数据复制到浮点向量中
    for (size_t i = 0; i < num_elements; ++i)
    {
        received_data[i] = ((float *)buffer)[i];
    }

    delete[] buffer;

    std::cout << "Received data: ";
    for (float f : received_data)
    {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    // 关闭套接字
    close(client_socket);
    close(server_fd);

    return received_data;
}

// 发送的算是写的差不多了
void send_tensor(int data, size_t num_elements)
{
    float *float_data = reinterpret_cast<float *>(data);
    std::cout << "Transfered Tensor Data from Python " << std::endl;
    for (size_t i = 0; i < num_elements; ++i)
    {
        std::cout << float_data[i] << " ";
    }
    std::cout << std::endl;
    _send(float_data, num_elements);
}

std::vector<float> recv_tensor(size_t num_elements)
{
    std::vector<float> data = _recv2(num_elements);
    return data;
}

PYBIND11_MODULE(tensor_transfer, m)
{
    m.doc() = "transmit tensor from python to c++";
    m.def("transfer_tensor", &transfer_tensor, "A function that receives a tensor from Python");
    m.def("send_tensor", &send_tensor, "A function that sends a tensor to Python");
    m.def("recv_tensor", &recv_tensor, "A function that receives a tensor from Python");
}
