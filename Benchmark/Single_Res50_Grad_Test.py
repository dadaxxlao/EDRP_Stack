import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
import time

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
          avg_grads_with_index = [(i, avg.item()) for i, avg in enumerate(avg_grads, start=0)]
      else:
          avg_grads_with_index = []

      # 单独处理最后一个块（如果存在）
      if last_chunk is not None:
          last_avg = torch.mean(last_chunk).item()
          avg_grads_with_index.append((len(self.grad_chunks) - 1, last_avg))

      avg_grads_with_index.sort(key=lambda x: abs(x[1]), reverse=True)

      # 其余代码保持不变
      top_30_percent_index_set = set(i for i, _ in avg_grads_with_index[:int(len(avg_grads_with_index) * 0.3)])
      high_grad_chunks = [self.grad_chunks[i] for i in top_30_percent_index_set]
      low_grad_chunks_indices = set(range(len(self.grad_chunks))) - top_30_percent_index_set
      low_grad_chunks = [self.grad_chunks[i] for i in low_grad_chunks_indices]

      return high_grad_chunks, low_grad_chunks

    def reconstruct_tensor(self, grad_chunks):
        sorted_chunks = sorted(grad_chunks, key=lambda x: x[0])
        reconstructed = torch.cat([chunk for _, chunk in sorted_chunks])
        return reconstructed

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),  # 因为ResNet50是针对224x224的图像设计的
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载CIFAR10数据集
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# 定义ResNet50模型
model = resnet50()
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10有10个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型转移到选择的设备上
model = model.to(device)
chunk_size = 1024
grad_merge = GradMerge(model)

total_gradmerge_time = 0
total_gradhandle_time = 0
    
# 训练模型
for epoch in range(1):  # 多次循环遍历数据集
    start_epoch_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)  # 将数据转移到相同的设备
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        start_gradmerge = time.time()

        merged_grads = grad_merge.merge_grads()

        start_gradhandle = time.time()
        #grad_handle = GradHandle(merged_grads, chunk_size)
        #high_grad_chunks, low_grad_chunks = grad_handle.process_grads()
        #reconstructed_tensor = grad_handle.reconstruct_tensor(high_grad_chunks + low_grad_chunks)
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

        running_loss += loss.item()
        if i % 100 == 99:   
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}')
            running_loss = 0.0
    end_epoch_time = time.time()
    total_gradmerge_time = total_gradmerge_time - total_gradhandle_time
    print("Ever Epoch Merge Time:", total_gradmerge_time, "Ever Epoch Handle Time:", total_gradhandle_time)
    print(f'Epoch {epoch + 1}:', end_epoch_time - start_epoch_time)

print('Finished Training')

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')