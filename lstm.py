# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:45:39 2020

@author: Zhe Cao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, cnn):
        super(Encoder, self).__init__()
        self.cnn = cnn
        
    def forward(self, x):
        enc_output = self.cnn(x)
        enc_output = F.relu(enc_output)
        return enc_output

    def freeze_bottom(self):
        for p in self.cnn.parameters():
            p.requires_grad = False
        for c in list(self.cnn.children())[-3:]:
            for p in c.parameters():
                p.requires_grad = True

    def freeze_all(self):
        for p in self.cnn.parameters():
            p.requires_grad = False
            
            
class Decoder(nn.Module):
    def __init__(self, enc_size, emb_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.enc_size = enc_size
        self.hidden_size = enc_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)
        self.lstm = nn.LSTM(input_size=self.emb_size+self.enc_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self._init_weights()

    def _init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, enc_out, captions, caplens):
        enc_out = enc_out.unsqueeze(0) # 1 * batch_size * 1000
        h0 = enc_out.cuda()
        c0 = enc_out.clone().cuda()
        captions = self.embed(captions) # seq_len * batch_size * emb_size
        enc_out = enc_out.repeat(captions.size(0), 1, 1) # seq_len * batch_size * 1000
        packed_captions = pack_padded_sequence(torch.cat((captions, enc_out), dim=2), caplens) # concatenate tokens and features
        outputs, _ = self.lstm(packed_captions, (h0, c0))
        outputs = self.fc(outputs[0])
        return outputs

    def greedy_pred(self, enc_out, pred_len):
        init = torch.ones((1, enc_out.size(0)), dtype=int) *  2
        init = init.cuda()
        enc_out = enc_out.unsqueeze(0) 
        h0 = enc_out.cuda()
        c0 = enc_out.clone().cuda()
        init = self.embed(init) # 1 * batch_size * emb_size
        next_out, (h, c) = self.lstm(torch.cat((init, enc_out), dim=2), (h0, c0))
        next_out = self.fc(next_out)
        next_in = next_out.argmax(dim=2)
        outputs = next_in.clone()
        for i in range(pred_len - 1):
            next_in = self.embed(next_in)
            next_out, (h, c) = self.lstm(torch.cat((next_in, enc_out), dim=2), (h, c))
            next_out = self.fc(next_out)
            next_in = next_out.argmax(dim=2)
            outputs = torch.cat((outputs, next_in), dim=0)
        return outputs

    def beam_search_pred(self, enc_outs, pred_len, beam_size=3):
        batch_size = enc_outs.size(0)
        outputs = torch.ones((pred_len, batch_size), dtype=int) # place_holder for outputs
        for idx in range(batch_size):
            enc_out = enc_outs[idx]
            enc_out = enc_out.unsqueeze(0).unsqueeze(0) # 1 * 1 * enc_size
            enc_out = enc_out.repeat(1, beam_size, 1) # 1 * beam_size * enc_size, (view beam_size as batch_size for convenience)
            k_words = torch.ones((1, beam_size), dtype=int).cuda() * 2 # 1 * beam_size
            seqs = k_words # 1 * beam_size
            k_scores = torch.zeros(1, beam_size, 1).cuda() # 1 * beam_size * 1
            h = enc_out.cuda() # 1 * beam_size * enc_size
            c = enc_out.clone().cuda() # 1 * beam_size * enc_size

            for step in range(pred_len):
                embedding = self.embed(k_words) # 1 * beam_size * emb_size
                lstm_out, (h, c) = self.lstm(torch.cat((embedding, enc_out), dim=2), (h, c))
                scores = self.fc(lstm_out) # 1 * beam_size * vocab_size
                scores =  F.log_softmax(scores, dim=2) # 1 * beam_size * vocab_size
                scores = k_scores.expand_as(scores) + scores # 1 * beam_size * vocab_size
                if step == 0: # first step
                    scores = scores.squeeze() # beam_size * vocab_size
                    k_scores, k_words = scores[0].topk(beam_size) # beam_size
                else:
                    scores = scores.squeeze() # beam_size * vocab_size
                    k_scores, k_words = scores.view(-1).topk(beam_size) # beam_size
                prev_idx = k_words / self.vocab_size # beam_size (between 0 and beam_size)
                next_idx = k_words % self.vocab_size # beam_size
                seqs = seqs[:, prev_idx] # L * beam_size
                seqs = torch.cat((seqs, next_idx.unsqueeze(0)), dim=0) # L * beam_size
                k_scores = k_scores.unsqueeze(0).unsqueeze(-1)
                k_words = next_idx.unsqueeze(0)
            output = seqs[:, k_scores.squeeze().argmax()]
            outputs[:, idx] = output[1:] # Don't include <sos>
        return outputs
            
            
