'''
对显存进行测试的过程
'''
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cProfile
import pstats
from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
import datetime
import time
from torch.distributed.elastic.multiprocessing.errors import record
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"{method.__name__} 运行时间: {te - ts} 秒")
        return result
    return timed

class GradMerge:
    def __init__(self, model):
        self.model = model
        self.grads = []
        self.shapes = []
        self.parameters = []
        self._register_hooks()

    def _save_grad(self, grad, param):
        self.grads.append(grad.view(-1))
        self.shapes.append(grad.shape)
        self.parameters.append(param)

    def _register_hooks(self):
        for param in self.model.parameters():
            # 使用lambda函数来保存对应的参数引用
            hook = lambda grad, param=param: self._save_grad(grad, param)
            param.register_hook(hook)

    def merge_grads(self):
        return torch.cat(self.grads)

    def split_grads(self, merged_grads):
        split_grads = []
        index = 0
        for shape in self.shapes:
            numel = torch.tensor(shape).prod().item()
            split_grads.append(merged_grads[index:index + numel].view(shape))
            index += numel
        return split_grads

    def update_model_grads(self, split_grads):
        assert len(split_grads) == len(self.parameters)
        for param, grad in zip(self.parameters, split_grads):
            param.grad = grad
            
#ATTENTION: GPU VERSION
class GradHandle:
    
    def __init__(self, grad_tensor, chunk_byte):
        self.grad_tensor = grad_tensor
        self.chunk_byte = chunk_byte
        self.chunk_size = self.chunk_byte // self.grad_tensor.element_size()
        print("Chunk size", self.chunk_size)
        self.grad_chunks = []
        self._split_tensor()
    
    def _split_tensor(self):
          num_chunks = (self.grad_tensor.numel() + self.chunk_size - 1) // self.chunk_size
          print("Num chunks", num_chunks)
          chunks = torch.chunk(self.grad_tensor, num_chunks)
          self.grad_chunks = list(enumerate(chunks))
          return self.grad_chunks
    
    def process_grads(self):

      # 分离最后一个块
      last_chunk = self.grad_chunks[-1][1] if self.grad_chunks else None
      other_chunks = [chunk for _, chunk in self.grad_chunks[:-1]]

      # 批量处理前n-1个块
      if other_chunks:
          stacked_chunks = torch.stack(other_chunks)
          avg_grads = torch.mean(stacked_chunks, dim=1)
      else:
          avg_grads = torch.tensor([]).to("cuda:0")  # 如果没有其它块，则创建一个空的Tensor

      # 单独处理最后一个块（如果存在）
      if last_chunk is not None:
          last_avg = torch.tensor([torch.mean(last_chunk)]).to("cuda:0")
          avg_grads = torch.cat((avg_grads, last_avg))

      # 使用 torch.topk 获取具有最大平均梯度值的块的索引
      num_top_chunks = int(len(self.grad_chunks) * 0.3)
      topk_values, topk_indices = torch.topk(avg_grads.abs(), num_top_chunks)

      # 根据索引获取高梯度块
      high_grad_chunks = [self.grad_chunks[i] for i in topk_indices]

      # 获取剩余的低梯度块
      low_grad_chunks_indices = set(range(len(self.grad_chunks))) - set(topk_indices.tolist())
      low_grad_chunks = [self.grad_chunks[i] for i in low_grad_chunks_indices]

      return high_grad_chunks, low_grad_chunks
    
    def reconstruct_tensor(self, grad_chunks):
        sorted_chunks = sorted(grad_chunks, key=lambda x: x[0])
        reconstructed = torch.cat([chunk for _, chunk in sorted_chunks])
        return reconstructed

def custom_all_reduce(tensor):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buffer = tensor.clone()
    recv_buffer = torch.zeros_like(tensor)
    for i in range(size - 1):
        send_rank = (rank + 1) % size
        recv_rank = (rank - 1 + size) % size

        # Initiate asynchronous send and receive.
        if (rank % 2 == 0):
            send_req = dist.isend(send_buffer, send_rank)
            recv_req = dist.irecv(recv_buffer, recv_rank)
        else:
            recv_req = dist.irecv(recv_buffer, recv_rank)
            send_req = dist.isend(send_buffer, send_rank)

        # Wait for the asynchronous calls to complete.
        send_req.wait()
        recv_req.wait()

        # Accumulate the received tensor.
        tensor += recv_buffer

        # Prepare the next round of send.
        send_buffer = recv_buffer.clone()

    return tensor

def average_gradients_gradOP(split_grads):
    world_size = float(dist.get_world_size())
    
    # 遍历 split_grads 列表中的每个元组（索引和张量）
    for idx, grad_chunk in split_grads:
        reduced_grad = custom_all_reduce(grad_chunk)
        averaged_grad = reduced_grad / world_size
        #dist.all_reduce(grad_chunk, op=dist.ReduceOp.SUM)
        #averaged_grad = grad_chunk / world_size
        split_grads[idx] = (idx, averaged_grad)

    return split_grads
     
def average_gradients_merge(merged_grads):
    size = float(dist.get_world_size())
    reduced_grad = custom_all_reduce(merged_grads)
    reduced_grad = reduced_grad / size
    return reduced_grad
   
def gpu_mem_usage():
    print("当前分配的显存（M）:", torch.cuda.memory_allocated()/1024/1024)
    print("当前保留的显存总量（M）:", torch.cuda.memory_reserved()/1024/1024)

n = 10000000 # 张量的大小
tensor = torch.randn(n).to("cuda:0")
print("Tensor", tensor)

def run():
    env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    print("Initialized process group")
    chunk_size = 2**20
    grad_handle = GradHandle(tensor, chunk_size)
    high_grad_chunks, low_grad_chunks = grad_handle.process_grads()
    split_grads = grad_handle._split_tensor()
    avg_split_grads = average_gradients_gradOP(split_grads)
    reconstructed_tensor = grad_handle.reconstruct_tensor(avg_split_grads)

'''
with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 监控 CPU 和 CUDA (GPU) 活动
        record_shapes=True,
        with_stack=True,  # 记录调用堆栈
        profile_memory=True,  # 监控内存分配
    ) as prof:
'''
    
    # 结束 Profiler 上下文
    # 输出 Profiler 结果
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # 保存 Profiler 结果到文件
    #prof.export_chrome_trace("./trace.json")
    #dist.destroy_process_group()
    
run()