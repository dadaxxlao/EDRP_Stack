import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
import datetime
import time
from torch.distributed.elastic.multiprocessing.errors import record
import torchvision.models as models
import matplotlib.pyplot as plt

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
    def clear_grads(self):
        self.grads.clear()
        self.shapes.clear()
        self.parameters.clear()
            
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
    
    def clear_grad_chunks(self):
        self.grad_chunks.clear()

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)  # CIFAR10 has 10 classes

    def forward(self, x):
        return self.model(x)

def partition_dataset():
    """ Partitioning CIFAR10 """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for GoogleNet
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        '../data',
        train=True,
        download=True,
        transform=transform)
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz

def test_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the model on the test set: {:.2f}%'.format(accuracy))
    return accuracy

def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform),
        batch_size=1000, shuffle=True)
    return test_loader

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

def average_gradients(model):
    """ Gradient averaging with custom all_reduce. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is None:
            continue
        reduced_grad = custom_all_reduce(param.grad.data)
        param.grad.data = reduced_grad / size
        


def average_gradients_merge(merged_grads):
    size = float(dist.get_world_size())
    reduced_grad = custom_all_reduce(merged_grads)
    reduced_grad = reduced_grad / size
    return reduced_grad

def average_gradients_gradOP(split_grads):
    world_size = float(dist.get_world_size())
    for idx, grad_chunk in split_grads:
        # 使用 custom_all_reduce 对每个张量分块进行全局累加
        reduced_grad = custom_all_reduce(grad_chunk)
        averaged_grad = reduced_grad / world_size
        #dist.all_reduce(grad_chunk, op=dist.ReduceOp.SUM)
        #averaged_grad = grad_chunk / world_size
        split_grads[idx] = (idx, averaged_grad)

    return split_grads

def gpu_mem_usage():
    print("当前分配的显存（M）:", torch.cuda.memory_allocated()/1024/1024)
    print("当前保留的显存总量（M）:", torch.cuda.memory_reserved()/1024/1024)
    
def run(rank, size, chunk_byte):
    """ Distributed Synchronous SGD Example """
    torch.cuda.empty_cache()
    device = torch.device("cuda:{}".format(0))
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    print("Train Data Download and Partition Finished")
    model = Net().to(device)
    print("Model Set Finished")
    grad_merge = GradMerge(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    print("Entering Train Main Loop")
    
    total_start_time = time.time()  # 总运行时间计时开始
    for epoch in range(1):
        epoch_loss = 0.0
        data_processed = 0  # 已处理数据计数器
        grad_handle_time = 0  # 梯度处理时间
        iteration_total_time = 0
        comm_time = 0  # 通信时间
        gpu_mem_usage()
        for data, target in train_set:
            
            iteration_start_time = time.time()  # 当前批次开始时间
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            epoch_loss += loss
            loss.backward()
            
            #Grad Operation
            grad_process_start = time.time()
            gpu_mem_usage()
            merged_grads = grad_merge.merge_grads()
            grad_handle = GradHandle(merged_grads, chunk_byte)
            high_grad_chunks, low_grad_chunks = grad_handle.process_grads()
            split_grads = grad_handle._split_tensor()
            
            #communication
            comm_start = time.time()
            avg_split_grads = average_gradients_gradOP(split_grads)
            comm_time += time.time() - comm_start
            
            #reconstruct
            reconstructed_tensor = grad_handle.reconstruct_tensor(avg_split_grads)
            original_grads = grad_merge.split_grads(reconstructed_tensor)
            grad_merge.update_model_grads(original_grads)
            
            grad_handle_time += time.time() - grad_process_start

            optimizer.step()
            
            grad_merge.clear_grads()
            grad_handle.clear_grad_chunks()
            
            iteration_time = time.time() - iteration_start_time
            iteration_total_time += iteration_time
            data_processed = data_processed + 1 
            if data_processed % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {data_processed + 1}, Loss: {loss.item()}, culmulative time: {iteration_total_time}')
            
         
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches, "epoch time: ", 
              iteration_total_time, "grad handle time: ", 
              grad_handle_time - comm_time, "comm time: ", 
              comm_time)
        
        
    total_end_time = time.time() - total_start_time  # 总运行时间计时结束
    print("Total time:", total_end_time)
    return total_end_time
    #test_loader = get_test_loader()
    #test_model(model, device, test_loader)


if __name__ == "__main__":
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    print("Initialized process group")
    chunk_bytes = [2**exp for exp in range(15, 21)]  # 生成2**10到2**20的chunk_byte值
    total_times = []

    for chunk_byte in chunk_bytes:
        total_time = run(rank, size, chunk_byte)
        total_times.append(total_time)
        
    plt.plot(chunk_bytes, total_times)
    plt.xlabel('Chunk Byte')
    plt.ylabel('Total End Time (s)')
    plt.title('Total End Time vs Chunk Byte')
    plt.xscale('log')  # 因为chunk_byte值变化范围很大，使用对数尺度
    plt.savefig('total_time_vs_chunk_byte.png')
    dist.destroy_process_group()
