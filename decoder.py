from transformer import CasualMaskTransformer, PositionalEncoding
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, num_embed, d_embed, d_qk, d_v, d_model, n_att, n_transformers, max_length, num_vocab):
        super().__init__()
        assert d_model == d_embed

        self.embedding = nn.Embedding(num_embed, d_embed)
        self.transformers = nn.ModuleList([CasualMaskTransformer(d_model, d_qk, d_v, n_att, max_length) for _ in range(n_transformers)])
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.lin = nn.Linear(d_model, num_vocab)

    def forward(self, x, pad_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        for transformer in self.transformers:
            x = transformer(x, pad_mask)

        return self.lin(x)
    