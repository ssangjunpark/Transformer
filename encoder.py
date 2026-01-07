from utils import Transformer, PositionalEncoding
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_embed, d_embed, d_qk, d_v, d_model, n_att, n_transformers, max_length, n_classes):
        super().__init__()
        assert d_model == d_embed

        self.embedding = nn.Embedding(num_embed, d_embed)
        self.transformers = nn.ModuleList([Transformer(d_model, d_qk, d_v, n_att) for _ in range(n_transformers)])
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.lin = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for block in self.transformers:
            x = block(x, mask=mask)

        x = x[:, 0, :] # since N x T x D 
        x = self.lin(x)
        return x