#Chunk size和时间的图，进行了benchmark
import torch
import time
import cProfile
import pstats
import matplotlib.pyplot as plt
import time
# 定义 GradHandle 类
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

# 生成随机数一维张量
n = 100000000  # 张量的大小
tensor = torch.randn(n).to("cuda:0")

def test_grad_handle(tensor, chunk_size):
    
    # 创建 GradHandle 实例
    #chunk_size = 8
    grad_handle = GradHandle(tensor, chunk_size)

    # 输出分割后的小张量列表
    split_grad_chunks = grad_handle.grad_chunks
    #print("Split grad", split_grad_chunks)

    # 计算平均梯度并排序
    high_grad_chunks, low_grad_chunks = grad_handle.process_grads()
    #print("high_grad_chunks len: ", len(high_grad_chunks))
    #print("low_grad_chunks len: ", len(low_grad_chunks))

    #high_grad_avg = torch.mean(torch.abs(torch.cat([chunk for _, chunk in high_grad_chunks]))).item()
    #low_grad_avg = torch.mean(torch.abs(torch.cat([chunk for _, chunk in low_grad_chunks]))).item()
    #print("Average absolute value of tensors in high_grad_chunks:", high_grad_avg)
    #print("Average absolute value of tensors in low_grad_chunks:", low_grad_avg)

    # 重构张量
    reconstructed_tensor = grad_handle.reconstruct_tensor(high_grad_chunks + low_grad_chunks)
    #print("Reconstrcuted", reconstructed_tensor)

def performance_test():
    chunk_sizes = list(range(128, 8193, 200))
    average_execution_times = []

    for size in chunk_sizes:
        total_time = 0
        num_tests = 5

        for _ in range(num_tests):
            start_time = time.time()
            test_grad_handle(tensor, size)
            end_time = time.time()
            total_time += (end_time - start_time)

        average_time = total_time / num_tests
        average_execution_times.append(average_time)

    return chunk_sizes, average_execution_times

# 执行性能测试并获取结果
chunk_sizes, average_execution_times = performance_test()

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(chunk_sizes, average_execution_times, marker='o')
plt.xlabel('Chunk Size')
plt.ylabel('Average Execution Time (seconds)')
plt.title('Average Performance Test for Different Chunk Sizes')
plt.axhline(y=5, color='r', linestyle='--')
plt.ylim(0, 6)  # 调整 y 轴范围以更好地关注 5 秒以内的性能
plt.grid(True)
plt.savefig("performance_chart_gpu.png")
plt.show()

# 打印表格数据
print("Chunk Size | Average Execution Time (seconds)")
print("---------------------------------------------")
for size, time in zip(chunk_sizes, average_execution_times):
    print(f"{size:<11} | {time:.2f}")