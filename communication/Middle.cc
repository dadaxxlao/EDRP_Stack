#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <stdio.h>

namespace py = pybind11;

size_t GL_chunk_ele_num, GL_total_chunk_num, GL_grad_shape, GL_Last;
std::vector<std::pair<int, float*>> IM_vec;
std::vector<std::pair<int, float*>> UN_vec;

void init(size_t grad_shape, size_t chunk_size){
    GL_grad_shape = grad_shape;
    GL_chunk_ele_num = chunk_size / 4; //每一个分块所包含的数量
    GL_total_chunk_num = (grad_shape + GL_chunk_ele_num - 1) / GL_chunk_ele_num; //总共的分块数量
    GL_Last = grad_shape % GL_chunk_ele_num; //最后一个剩余的分块大小(Number)
}

void tensor_trans(uintptr_t ptr, int index, int IM){
    
    float *tensor = reinterpret_cast<float *>(ptr);
    if (IM){
        //std::cout << "Important Tensor Received" << std::endl;
        IM_vec.push_back(std::make_pair(index, tensor));
        std::cout << "Received tensor elements: " << std::endl;
        printf("Address: %p\n", tensor);
        for (size_t i = 0; i < GL_chunk_ele_num; ++i)
        {
        std::cout << tensor[i] << " ";
        }
        std::cout << std::endl;
    }
    else {
        //std::cout << "Unimportant Tensor Received" << std::endl;
        UN_vec.push_back(std::make_pair(index, tensor));
        std::cout << "Received tensor elements: " << std::endl;
        for (size_t i = 0; i < GL_chunk_ele_num; ++i)
        {
        std::cout << tensor[i] << " ";
        }
        std::cout << std::endl;
    }
}

void verify(){
    //打印IM_vec和UN_vec的内容
    std::cout << "Print High Grad \n" << std::endl;
    for (uint64_t i = 0; i < IM_vec.size(); i++){
        std::cout << "Index: " << IM_vec[i].first << std::endl;
        float *tensor = IM_vec[i].second;
        printf("Address: %p\n", tensor);
        for (size_t j = 0; j < GL_chunk_ele_num; ++j){
            std::cout << tensor[j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Print Low Grad \n" << std::endl;
    for (uint64_t i = 0; i < UN_vec.size(); i++){
        std::cout << "Index: " << UN_vec[i].first << std::endl;
        for (size_t j = 0; j < GL_chunk_ele_num; ++j){
            std::cout << UN_vec[i].second[j] << " ";
        }
        std::cout << std::endl;
    }
}

PYBIND11_MODULE(MIDDLE, m)
{
    m.doc() = "Middle Module between Pytorch and C++";
    m.def("init", &init, "A function that receives a tensor from Python");
    m.def("tensor_trans", &tensor_trans, "A function that sends a tensor to Python");
    m.def("verify", &verify, "A function that sends a tensor to Python");
}