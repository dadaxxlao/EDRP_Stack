import torch
import os
import tensor_transfer
import ctypes

tensor = torch.randn(100)
type = tensor.dtype
print("python side", tensor, "type", type)
ptr  = tensor.data_ptr()
c_ptr = ctypes.c_void_p(ptr)
print("Python pointer address:", ctypes.cast(ptr, ctypes.c_void_p))
print("ptr", ptr)
num_elements = tensor.numel()
print("num_elements", num_elements)

tensor_transfer.receive_tensor(ptr, num_elements)
