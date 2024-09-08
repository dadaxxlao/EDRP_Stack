'''
这是对数据分区性能的测试
将数据集切分所花费的时间
'''
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
        print("Data length", data_len)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset():
    """ Partitioning CIFAR10 """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        '../data',
        train=True,
        download=True,
        transform=transform)
    size = 4 # world size
    bsz = 1024 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(0)
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz

start_time = time.time()
train_set, bsz = partition_dataset()
print("Train set Length", len(train_set))
print("Batch size", bsz)
end_time = time.time()
print("Partition time: ", end_time - start_time)