import torch as th
import torch.nn.functional as F
from torch import nn
import math


class MultiLinear(nn.Module):
    def __init__(self, shape, equation, dtype=th.float32, bias_shape=None):
        super().__init__()
        self.weight = nn.Parameter(th.empty(shape, dtype=dtype))
        self.bias = nn.Parameter(th.empty(bias_shape, dtype=dtype))
        self.equation = equation
        for param in self.parameters():
            nn.init.xavier_normal_(param)
        
    def forward(self, input):
        return th.einsum(self.equation, input, self.weight) + self.bias

@th.jit.script
class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, device=th.device('cuda')):
        super().__init__()
        self.device = device
        self.output_gate = MultiLinear((hidden_size, input_size, embedding_dim), 'bi, lij -> blj', bias_shape=(hidden_size, embedding_dim)).to(self.device)
        self.forget_gate = nn.Linear(input_size, hidden_size).to(self.device)
        self.input_gate = nn.Linear(input_size, hidden_size).to(self.device)
        self.value = MultiLinear((hidden_size, input_size, embedding_dim), 'bi, lij -> blj', bias_shape=(hidden_size, embedding_dim)).to(self.device)
        self.key = MultiLinear((hidden_size, input_size, embedding_dim), 'bi, lij -> blj', bias_shape=(hidden_size, embedding_dim)).to(self.device)
        self.query = MultiLinear((hidden_size, input_size, embedding_dim), 'bi, lij -> blj', bias_shape=(hidden_size, embedding_dim)).to(self.device)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.scale = math.sqrt(hidden_size)
        
        self.hidden_state = None
        self.cell_state = None
        self.m = None
        self.n = None
        
    @th.jit.export
    def reset_states(self, batch_size: int):
        batch_size = int(batch_size)
        self.hidden_state, self.cell_state = (th.zeros((batch_size, self.hidden_size, self.embedding_dim), dtype=th.float32).to(self.device), 
                                              th.zeros((batch_size, self.hidden_size, self.embedding_dim, self.embedding_dim), dtype=th.float32).to(self.device))
        self.m, self.n = (th.zeros((batch_size, self.hidden_size), dtype=th.float32).to(self.device), 
                          th.zeros((batch_size, self.hidden_size, self.embedding_dim), dtype=th.float32).to(self.device))
        
    @th.jit.export
    def _forward_step(self, input):
        o = F.sigmoid(self.output_gate(input))
        f_ = self.forget_gate(input)
        i_ = self.input_gate(input)
        v = self.value(input)
        k = self.key(input)/self.scale
        q = self.query(input)
        m = th.amax(th.stack((f_ + self.m, i_), dim=-1), dim=-1)
        i = th.exp(i_ - m)
        f = th.exp(f_ + self.m - m)
        self.m = m
        self.cell_state = (f.unsqueeze(-1).unsqueeze(-1)*self.cell_state) + i.unsqueeze(-1).unsqueeze(-1)*th.einsum('bli, blj -> blij', v, k)
        self.n = f.unsqueeze(-1)*self.n + i.unsqueeze(-1)*k
        self.hidden_state = o*th.einsum('blij, bli -> blj', self.cell_state, q)/th.clamp(th.sum(self.n*q, dim=-1), 1, th.inf).unsqueeze(-1)
    
    def forward(self, input):
        input = input.to(self.device)
        self.reset_states(len(input))
        for t in range(input.shape[1]):
            self._forward_step(input[:, t, :])
        return self.hidden_state, self.cell_state
