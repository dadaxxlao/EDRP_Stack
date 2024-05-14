'''
这部分是用来完成对中间层的实现
中间层上接GradHandle，下接Socket通信过程
主要作用是用来传递信息，和分别调用API

初始化时只需要获得当前Grad的信息，即分块后的Grad大小，数量，总Grad大小
那么我们就可以确定Middle部分的功能了：
 1. 初始化：接收合并后梯度的形状，还有相应的分块大小。
 2. 发送功能：传入两个列表（High 和 Low），调用底层DPDK的接口 发送目标
 3. 接收功能：返回接收到的整个的梯度 接受目标
'''
import MIDDLE


class Middle:
    
    def __init__(self, grad_shape, chunk_size):
        self.grad_shape = grad_shape #grad_shape是指梯度的形状,长度
        self.chunk_size = chunk_size #chunk_size是指每个块的大小(Byte)
        self.chunk_ele_num = self.chunk_size // 4 #每个块的元素个数
        self.total_chunk_num = (self.grad_shape + self.chunk_ele_num - 1) // self.chunk_ele_num
        print("Total Chunk Num", self.total_chunk_num)
        print("chunk element number", self.chunk_ele_num)
        MIDDLE.init(self.grad_shape, self.chunk_size)
        '''
        剩下的就是完成C++侧的API，调用该API传递这两个变量即可
        '''
    def send(self, high_grad_chunks, low_grad_chunks):
        for index, high_chunk in high_grad_chunks:
            #print(f"High Grad Chunk {index}: {high_chunk}")
            ptr = high_chunk.data_ptr()
            MIDDLE.tensor_trans(ptr, index, 1)
            
        for index, low_chunk in low_grad_chunks:
            # 对每个低梯度块执行相似的操作
            #print(f"Low Grad Chunk {index}: {low_chunk}")
            ptr = low_chunk.data_ptr()
            MIDDLE.tensor_trans(ptr, index, 0)
            # 调用C++侧的低级别发送API 完成发送
    def recv(self):
        '''
        在C++侧接收为数组或者Vector
        data_from_cpp = tensor_transfer.recv_tensor(num_elements)
        tensor = torch.tensor(data_from_cpp)
        '''
        MIDDLE.verify()
        #Use C++ API to receive the whole Merged grad
        