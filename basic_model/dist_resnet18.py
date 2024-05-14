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
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # CIFAR10 has 10 classes

    def forward(self, x):
        return self.model(x)



def partition_dataset():
    """ Partitioning CIFAR10 """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        './data',
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
    start_time = time.time()
    for param in model.parameters():
        if param.grad is None:
            continue
        # Custom all_reduce function is called and the result is stored in reduced_grad.
        reduced_grad = custom_all_reduce(param.grad.data)
        # dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        # Updating the gradient with the averaged values.
        param.grad.data = reduced_grad / size
        #param.grad.data /= size
    end_time = time.time()
    return end_time - start_time

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    total_comm_time = 0  # Total communication time
    
    device = torch.device("cuda:{}".format(0))
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    print("Train Data Download and Partition Finished")
    #model = Net()
    #model = model
    #model = model.cuda(rank)
    model = Net().to(device)
    print("Model Set Finished")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    start_time = datetime.datetime.now()
    print("Training started at", start_time)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    print("Entering Train Main Loop")
    for epoch in range(5):
        #comp--compution time
        #comm--communication time
        comp_start_time = time.time()
        comm_time = 0
        epoch_loss = 0.0
        for data, target in train_set:
#            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss
            loss.backward() #这时候已经得出了所有的梯度
            comm_time += average_gradients(model)
            optimizer.step()
        comp_end_time = time.time()
        total_comm_time += comm_time
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches, ', epoch time: ',
              comp_end_time - comp_start_time, 
              ', total comm time: ', total_comm_time, 
              'Commtime per epoch: ', total_comm_time/(comp_start_time - comp_end_time))
    # After the training loop
    end_time = datetime.datetime.now()
    print("Training finished at", end_time)
    print("Total training time: ", end_time - start_time)
    #test_loader = get_test_loader()
    #test_model(model, device, test_loader)


if __name__ == "__main__":
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    os.environ["NCCL_AVOID_RECORD_STREAMS"] = "0"
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    print("Initialized process group")
    run(rank, size)
    dist.destroy_process_group()
