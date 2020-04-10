# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:24:49 2020

@author: Zhe Cao
"""
import math
import torch
import torch.nn as nn
import torchtext
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class Encoder(nn.Module):
    def __init__(self, cnn, channels, embed_size):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(*list(cnn.children())[:-2])
        self.channels = channels
        self.conv1 = nn.Conv2d(self.channels, embed_size, 1)
        self.embed_size = embed_size
        
    def forward(self, x):
        enc_output = self.cnn(x) # batch_size * 2048 * h * w
        batch_size, _, _, _ = enc_output.shape
        enc_output = self.conv1(enc_output)
        enc_output = enc_output.view(batch_size, self.embed_size, -1)
        enc_output = enc_output.permute(2, 0, 1) # hw * batch_size * 300(embed size)

        return enc_output

    def freeze_bottom(self):
        for p in self.cnn.parameters():
            p.requires_grad = False
        for c in list(self.cnn.children())[-2:]: # Only train the last two blocks
            for p in c.parameters():
                p.requires_grad = True

    def freeze_all(self):
        for p in self.cnn.parameters():
            p.requires_grad = False
            
            
class PositionEncoder(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, nHead, nHid, nLayers, dropout_dec=0.5, dropout_pos=0.1, use_pretrained_embed=False, embedding_matrix=None):
        super(Decoder, self).__init__()
        self.src_mask = None
        self.pos_decoder = PositionEncoder(emb_size, dropout_pos)
        decoder_layers = TransformerDecoderLayer(emb_size, nHead, nHid, dropout_dec)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nLayers)
        self.emb_size = emb_size
        if use_pretrained_embed:
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, emb_size)
        self.decode = nn.Linear(emb_size, vocab_size)
    
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        self.decode.bias.data.zero_()
        self.decode.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, memory):
        src = src.permute(1,0)  # src*batch size
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = self._generate_square_subsequent_mask(len(src)).cuda()

        src = self.embed(src)   # src*batch*size
        src = src*math.sqrt(self.emb_size)  
        src = self.pos_decoder(src)
        output = self.transformer_decoder(src, memory, self.src_mask)
        output = self.decode(output)
        return output

    def pred(self, memory, pred_len):
        batch_size = memory.size(1)
        src = torch.ones((pred_len, batch_size), dtype=int) * 2
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = self._generate_square_subsequent_mask(len(src)).cuda()
        output = torch.ones((pred_len, batch_size), dtype=int)
        src, output = src.cuda(), output.cuda()
        for i in range(pred_len):
            src_emb = self.embed(src) # src_len * batch size * embed size
            src_emb = src_emb*math.sqrt(self.emb_size)
            src_emb = self.pos_decoder(src_emb)
            out = self.transformer_decoder(src_emb, memory, self.src_mask)
            out = out[i]
            out = self.decode(out) # batch_size * vocab_size
            out = out.argmax(dim=1)
            if i < pred_len-1:
                src[i+1] = out
            output[i] = out
        return output
    
    