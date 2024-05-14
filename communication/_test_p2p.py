"""
这是对C++中Socket进行Tensor传输的整体测试
我们的下一步是在分布式训练中完成中间层
用来完成整个梯度交换的过程 并将底层所需要的信息传递给C++ Socket
"""
import torch
import torch.distributed as dist
import os
import tensor_transfer

def self_send(tensor):
    tensor = tensor.to("cpu")
    ptr = tensor.data_ptr()
    num_elements = tensor.numel()
    tensor_transfer.send_tensor(ptr, num_elements)

def self_recv(num_elements):
    data_from_cpp = tensor_transfer.recv_tensor(num_elements)
    tensor = torch.tensor(data_from_cpp)
    print("tensor", tensor)
    return tensor

    

def main(rank, size):
    # 初始化进程组
    dist.init_process_group(backend='nccl', rank=rank, world_size=size)
    
    tensor =  torch.randn(100)
    num_elements = tensor.numel()
    
    print(f"Rank {rank} starting with tensor: {tensor}")

    if rank == 0:
        # Rank 0 发送数据给 Rank 1
        print("Here is Rank 0")
        self_send(tensor)
    elif rank == 1:

        print("Here is Rank 1")
        recv_tensor = self_recv(num_elements)
        print(f"Rank {rank} received tensor: {recv_tensor}")

    

    dist.destroy_process_group()

if __name__ == "__main__":

    rank = int(os.environ["RANK"])
    size = 2
    main(rank, size)
