'''
用来测试整体的grad_handle和grad_merge的时间
在单机进行测试
grad_merge的性能很好
grad_handle性能一般
具体grad_handle性能可看./Benchmark/Grad_Handle_Figure.py
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import time
import cProfile

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

def load_cifar10(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader

# 修改后的训练函数
def train(model, device, train_loader, optimizer, grad_merge, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_train_time = 0
    total_gradmerge_time = 0
    total_gradhandle_time = 0
    chunk_size = 1024

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            start_gradmerge = time.time()

            merged_grads = grad_merge.merge_grads()

            start_gradhandle = time.time()
            grad_handle = GradHandle(merged_grads, chunk_size)
            high_grad_chunks, low_grad_chunks = grad_handle.process_grads()
            reconstructed_tensor = grad_handle.reconstruct_tensor(high_grad_chunks + low_grad_chunks)
            end_gradhandle = time.time()

            original_grads = grad_merge.split_grads(merged_grads)
            grad_merge.update_model_grads(original_grads)

            end_gradmerge = time.time()

            total_gradmerge_time += end_gradmerge - start_gradmerge
            total_gradhandle_time += end_gradhandle - start_gradhandle

            grad_merge.grads = []
            grad_merge.shapes = []
            grad_merge.parameters = []

            optimizer.step()

            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}')
    total_gradmerge_time = total_gradmerge_time - total_gradhandle_time
    return total_train_time, total_gradmerge_time, total_gradhandle_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = load_cifar10()
    model = models.resnet50().to(device)
    grad_merge = GradMerge(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start_train = time.time()
    total_train_time, total_gradmerge_time, total_gradhandle_time = train(model, device, 
                                                                          train_loader, 
                                                                          optimizer, 
                                                                          grad_merge, 
                                                                          epochs=1)
    end_train = time.time()

    total_train_time = end_train - start_train

    print(f'Total training time: {total_train_time:.2f} seconds')
    print(f'Total GradMerge processing time: {total_gradmerge_time:.2f} seconds')
    print(f'Total GradHandle processing time: {total_gradhandle_time:.2f} seconds')

if __name__ == "__main__":
    main()  