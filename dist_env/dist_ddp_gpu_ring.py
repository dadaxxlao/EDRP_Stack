#!/usr/bin/env python

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
from torch.distributed.elastic.multiprocessing.errors import record


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
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


""" Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size
   print("Left: ", left)
   print("Right: ", right)
   print("send_buff_type: ", send_buff.dtype)
   print("Start Allreduce")

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           print("isend start")
           send_req = dist.isend(send_buff, right)
           print("isend finished & recv started")
           dist.recv(recv_buff, left)
           print("recv finished")
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           print("isend start")
           send_req = dist.isend(recv_buff, right)
           print("isend finished & recv started")
           dist.recv(send_buff, left)
           print("recv finished")
           accum[:] += send_buff[:]
       print("Start Wait")
       send_req.wait()
   recv[:] = accum[:]
   print("End Allreduce")
   
def isend_test(send,recv):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = torch.zeros(send.size())
    recv_buff = torch.zeros(send.size())
    accum = torch.zeros(send.size())
    accum[:] = send[:]
    torch.cuda.synchronize()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size
    print("Left: ", left)
    print("Right: ", right)
    print("send_buff_type: ", send_buff.dtype)
    print("send_buff: ", send_buff)
    print("isend start")
    send_req = dist.isend(tensor=send_buff, dst=right)
    print("isend finished & recv started")
    dist.recv(recv_buff, left)
    print("recv finished")
    accum[:] += recv_buff[:]
"""
def average_gradients(model):

    size = float(dist.get_world_size())
    for param in model.parameters():
        if type(param) is torch.Tensor:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            param.grad.data /= size    
"""

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        recv_buff = torch.zeros_like(param.grad.data)
        print("data type: ", param.grad.data.dtype)
        isend_test(param.grad.data, recv_buff)
        param.grad.data = recv_buff
        param.grad.data /= size 
            


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    device = torch.device("cuda:{}".format(0))
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    print("Train Data Download and Partition Finished")
    model = Net()
    #model = model
    #model = model.cuda(rank)
    model = Net().to(device)
    print("Model Set Finished")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    print("Entering Train Main Loop")
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
#            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

if __name__ == "__main__":
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    os.environ["NCCL_DEBUG"] = "ERROR"
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    print("Initialized process group")
    run(rank, size)
    dist.destroy_process_group()
