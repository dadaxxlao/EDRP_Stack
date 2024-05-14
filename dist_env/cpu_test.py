import torch
import torch.distributed as dist
import os
import argparse


def custom_all_reduce(tensor, rank):
    
    size = 2
    send_buffer = tensor.clone()
    recv_buffer = torch.zeros_like(tensor)
    print("send_buffer: ", send_buffer)
    for i in range(size - 1):
        send_rank = (rank + 1) % size
        recv_rank = (rank - 1 + size) % size

        # Initiate asynchronous send and receive.
        send_req = dist.isend(send_buffer, send_rank)
        recv_req = dist.irecv(recv_buffer, recv_rank)

        # Wait for the asynchronous calls to complete.
        send_req.wait()
        recv_req.wait()

        # Accumulate the received tensor.
        tensor += recv_buffer

        # Prepare the next round of send.
        send_buffer = recv_buffer.clone()

    return tensor

def init_process(rank, world_size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '192.168.11.150'  # Replace with the IP address of the master node
    os.environ['MASTER_PORT'] = '29500'        # Replace with the desired port
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def run(rank, world_size):
    """ The function to be executed on each node. """
    tensor = torch.rand(1).cuda(0)
    print('Rank ', rank, ' has data ', tensor)
    tensor = dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' computed ', tensor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()

    init_process(args.rank, args.world_size)
    run(args.rank, args.world_size)

if __name__ == "__main__":
    main()
