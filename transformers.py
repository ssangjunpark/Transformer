import math
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_model, d_qk, d_v):
        super().__init__()

        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v

        self.q = nn.Linear(self.d_model, self.d_qk)
        self.k = nn.Linear(self.d_model, self.d_qk)
        self.v = nn.Linear(self.d_model, self.d_v)
    
    def forward(self, x):
        q_out = self.q(x)
        k_out = self.k(x)
        v_out = self.v(x)
        
        scores = torch.matmul(q_out, k_out.transpose(-1, -2)) / math.sqrt(self.d_qk)
        attn = nn.functional.softmax(scores, dim=-1)

        return torch.matmul(attn, v_out)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_qk, d_v, n_att):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.n_att = n_att

        # project into ALL heads at once
        self.q = nn.Linear(d_model, d_qk * n_att)
        self.k = nn.Linear(d_model, d_qk * n_att)
        self.v = nn.Linear(d_model, d_v  * n_att)

        self.lin = nn.Linear(d_v * n_att, d_model)

    def forward(self, x, mask=None):
        N, T, _ = x.shape
        h = self.n_att

        q_out = self.q(x)
        k_out = self.k(x)
        v_out = self.v(x)

        q_out = q_out.view(N, T, h, self.d_qk).transpose(1, 2)
        k_out = k_out.view(N, T, h, self.d_qk).transpose(1, 2)
        v_out = v_out.view(N, T, h, self.d_v ).transpose(1, 2)

        scores = (q_out @ k_out.transpose(-2, -1)) / math.sqrt(self.d_qk)

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        out = torch.matmul(nn.functional.softmax(scores, dim=-1), v_out).transpose(1, 2).contiguous().view(N, T, h * self.d_v)

        return self.lin(out)
        
class Transformer(nn.Module):
    def __init__(self, d_model, d_qk, d_v, n_att):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v

        self.multiHeadAttention = MultiHeadAttention(d_model, self.d_qk, self.d_v, n_att)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.ANN = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.layerNorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layerNorm1(x + self.multiHeadAttention(x, mask=None))
        x = self.layerNorm2(x + self.ANN(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]