'''
这个版本是用来使用C++ socket进行通信
并且验证中间层的信息传递
可以尝试完成乱序接收的过程
先来完成将元组进行传递的功能和性能
'''

import torch
import time
import cProfile
import pstats
import Middle

#改为CPU版的
class GradHandle:
  
    def __init__(self, grad_tensor, chunk_byte):
        self.grad_tensor = grad_tensor
        self.chunk_byte = chunk_byte
        self.chunk_size = self.chunk_byte // self.grad_tensor.element_size()
        #print("Chunk size", self.chunk_size)
        self.grad_chunks = []
        self._split_tensor()
    
    def _split_tensor(self):
          num_chunks = (self.grad_tensor.numel() + self.chunk_size - 1) // self.chunk_size
          # 使用 torch.chunk 一次性分割张量
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
          avg_grads = torch.tensor([])  # 如果没有其它块，则创建一个空的Tensor

      # 单独处理最后一个块（如果存在）
      if last_chunk is not None:
          last_avg = torch.tensor([torch.mean(last_chunk)])
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
    
    def clear_grad_chunks(self):
        self.grad_chunks.clear()


def test_grad_handle():
    # 生成随机数一维张量
    n = 10  # 张量的大小
    tensor = torch.randn(n)
    print("Merged Tensor", tensor)

    # 创建 GradHandle 实例
    chunk_size = 8  # 小张量的大小
    grad_handle = GradHandle(tensor, chunk_size)
    Test = Middle.Middle(n, chunk_size)
    # 输出分割后的小张量列表
    split_grad_chunks = grad_handle.grad_chunks
    print("Split grad", split_grad_chunks)
    for index, chunk in split_grad_chunks:
        print(f"Chunk {index}: {chunk.data_ptr()}")
    
    # 计算平均梯度并排序
    high_grad_chunks, low_grad_chunks = grad_handle.process_grads()
    print("high_grad_chunks", high_grad_chunks)
    print("low_grad_chunks", low_grad_chunks)
    #print("high_grad_chunks len: ", len(high_grad_chunks))
    #print("low_grad_chunks len: ", len(low_grad_chunks))
    Test.send(high_grad_chunks, low_grad_chunks)
    Test.recv()

    #high_grad_avg = torch.mean(torch.abs(torch.cat([chunk for _, chunk in high_grad_chunks]))).item()
    #low_grad_avg = torch.mean(torch.abs(torch.cat([chunk for _, chunk in low_grad_chunks]))).item()
    #print("Average absolute value of tensors in high_grad_chunks:", high_grad_avg)
    #print("Average absolute value of tensors in low_grad_chunks:", low_grad_avg)

    # 重构张量
    reconstructed_tensor = grad_handle.reconstruct_tensor(high_grad_chunks + low_grad_chunks)
    #print("Reconstrcuted", reconstructed_tensor)
# 运行 cProfile 并保存到一个文件
#cProfile.run('test_grad_handle()', 'profiling_results')
test_grad_handle()
# 创建 pstats 对象并加载结果
#stats = pstats.Stats('profiling_results')

# 按照 "cumulative time"（累积时间）进行排序
# 'cumulative' 表示函数及其所有子函数的总运行时间
#stats.sort_stats('cumulative').print_stats()

# 如果你想按单个函数的时间进行排序，可以使用 'time'
# stats.sort_stats('time').print_stats()