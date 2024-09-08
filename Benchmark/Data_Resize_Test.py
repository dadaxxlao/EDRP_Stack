import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
import time

start_time = time.time()
# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),  # 因为ResNet50是针对224x224的图像设计的
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载CIFAR10数据集
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
end_time = time.time()
print("Load data time:", end_time - start_time)