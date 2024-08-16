import torch as th
import torch.nn.functional as F
from torch import nn
import math
from typing import Tuple, Union


class MultiLinear(nn.Module):
    def __init__(self, shape: Tuple[int], equation: str, bias_shape: Tuple[int, ...], dtype: th.dtype = th.float32):
        super().__init__()
        self.weight = nn.Parameter(th.empty(shape, dtype=dtype))
        self.bias = nn.Parameter(th.empty(bias_shape, dtype=dtype))
        self.equation = equation
        for param in self.parameters():
            nn.init.xavier_normal_(param)
        
    def forward(self, input: th.Tensor):
        return th.einsum(self.equation, input, self.weight) + self.bias

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, device=th.device('cuda')):
        super().__init__()
        self.device = device
        # output, value, key, query
        self.multilinear_gates = MultiLinear((hidden_size*4, input_size, embedding_dim), 'bi, lij -> blj', bias_shape=(hidden_size*4, embedding_dim)).to(self.device)
        # input, forget
        self.linear_gates = nn.Linear(input_size, hidden_size*2).to(self.device)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.scale = math.sqrt(embedding_dim)
        
        self.hidden_state, self.cell_state = (th.zeros((1, self.hidden_size, self.embedding_dim), dtype=th.float32).to(self.device), 
                                              th.zeros((1, self.hidden_size, self.embedding_dim, self.embedding_dim), dtype=th.float32).to(self.device))
        self.m, self.n = (th.zeros((1, self.hidden_size), dtype=th.float32).to(self.device), 
                          th.zeros((1, self.hidden_size, self.embedding_dim), dtype=th.float32).to(self.device))
        
    @th.jit.export
    def reset_states(self, batch_size: int):
        batch_size = int(batch_size)
        self.hidden_state, self.cell_state = (th.zeros((batch_size, self.hidden_size, self.embedding_dim), dtype=th.float32).to(self.device), 
                                              th.zeros((batch_size, self.hidden_size, self.embedding_dim, self.embedding_dim), dtype=th.float32).to(self.device))
        self.m, self.n = (th.zeros((batch_size, self.hidden_size), dtype=th.float32).to(self.device), 
                          th.zeros((batch_size, self.hidden_size, self.embedding_dim), dtype=th.float32).to(self.device))
        
    @th.jit.export
    def _forward_step(self, input: th.Tensor):
        linear_gates = self.linear_gates(input)
        multilinear_gates = self.multilinear_gates(input)
        o_, v, k, q = th.chunk(multilinear_gates, 4, dim=1)
        o = F.sigmoid(o_)
        i_, f_ = th.chunk(linear_gates, 2, dim=1)
        m = th.amax(th.stack((f_ + self.m, i_), dim=2), dim=2)
        i = th.exp(i_ - m)
        f = th.exp(f_ + self.m - m)
        self.m = m
        self.cell_state = (f.unsqueeze(-1).unsqueeze(-1)*self.cell_state) + i.unsqueeze(-1).unsqueeze(-1)*th.einsum('bli, blj -> blij', v, k)
        self.n = f.unsqueeze(-1)*self.n + i.unsqueeze(-1)*k
        self.hidden_state = o*th.einsum('blij, bli -> blj', self.cell_state, q)/th.clamp(th.sum(self.n*q, dim=-1), 1, th.inf).unsqueeze(-1)
    
    def forward(self, input: th.Tensor):
        input = input.to(self.device)
        self.reset_states(len(input))
        for t in range(input.shape[1]):
            self._forward_step(input[:, t, :])
        return self.hidden_state, self.cell_state

if __name__ == '__main__':
    model = xLSTM(17, 19, 23)
    compiled_model = th.jit.script(model)
    output = compiled_model(th.randn(7, 5, 17))
    print(output[0].shape, output[1].shape)
