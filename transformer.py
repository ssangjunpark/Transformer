import torch
import torch.nn as nn
from utils import Transformer as EncoderBlock
from utils import Seq2SeqDecoderBlock, PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, num_emb, d_model, d_qk, d_v, n_att, n_enc_blocks, n_dec_blocks, max_len):
        super().__init__()
        
        self.embedding = nn.Embedding(num_emb, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, d_qk, d_v, n_att) for _ in range(n_enc_blocks)
        ])
        
        self.decoder = nn.ModuleList([
            Seq2SeqDecoderBlock(d_model, d_qk, d_v, n_att, max_len) for _ in range(n_dec_blocks)
        ])
        
        self.fc = nn.Linear(d_model, num_emb)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        e_out = self.embedding(src)
        e_out = self.pos_enc(e_out)
        for block in self.encoder:
            e_out = block(e_out, mask=src_mask)
            
        d_out = self.embedding(tgt)
        d_out = self.pos_enc(d_out)
        for block in self.decoder:
            d_out = block(d_out, enc_out=e_out, pad_mask=tgt_mask, enc_mask=src_mask)
            
        return self.fc(d_out)

    def encode(self, src, src_mask=None):
        e_out = self.embedding(src)
        e_out = self.pos_enc(e_out)
        for block in self.encoder:
            e_out = block(e_out, mask=src_mask)
        return e_out
        
    def decode(self, tgt, enc_out, tgt_mask=None, enc_mask=None):
        d_out = self.embedding(tgt)
        d_out = self.pos_enc(d_out)
        for block in self.decoder:
            d_out = block(d_out, enc_out=enc_out, pad_mask=tgt_mask, enc_mask=enc_mask)
        return self.fc(d_out)
