# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:49:02 2020

@author: Zhe Cao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, cnn):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(*list(cnn.children())[:-2])
        
    def forward(self, x):
        enc_output = self.cnn(x) # batch_size * 2048 * h * w
        enc_output = enc_output.permute(0, 2, 3, 1) # batch_size * h * w * 2048
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
            
            
class Attention(nn.Module):
    def __init__(self, attn_size, dec_h_size, enc_size):
        super(Attention, self).__init__()
        self.enc_attn = nn.Linear(enc_size, attn_size)
        self.dec_attn = nn.Linear(dec_h_size, attn_size)
        self.V = nn.Linear(attn_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc_out, dec_h):
        """
        enc_out: batch_size * hw * enc_sie (2048)
        dec_out: batch_size * dec_h_size
        """
        attn1 = self.enc_attn(enc_out) # batch_size * hw * attn_size
        attn2 = self.dec_attn(dec_h) # batch_size * attn_size
        attn2 = attn2.unsqueeze(1) # batch_size * 1 * attn_size
        score = F.relu(attn1 + attn2) # batch_size * hw * attn_size
        alpha = self.softmax(self.V(score).squeeze(2)) # batch_size * hw
        context_vec = alpha.unsqueeze(2) * enc_out # batch_size * hw * enc_size
        context_vec = context_vec.sum(dim=1) # batch_size * enc_size
        return context_vec, alpha
    
    
class Decoder(nn.Module):
    def __init__(self, enc_size, emb_size, vocab_size, attn_size):
        super(Decoder, self).__init__()
        self.enc_size = enc_size # 2048
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.attn_size = attn_size
        self.hidden_size = enc_size
        
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)
        self.attention = Attention(self.attn_size, self.hidden_size, self.enc_size)
        self.lstm_cell = nn.LSTMCell(input_size=self.enc_size+self.emb_size, hidden_size=self.hidden_size)
        self.fc_h = nn.Linear(self.enc_size, self.hidden_size)
        self.fc_c = nn.Linear(self.enc_size, self.hidden_size)
        self.fc_gate = nn.Linear(self.hidden_size, self.enc_size) # Additional gate
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, enc_out, captions, caplens):
        captions = self.embed(captions) # batch_size * seq_len * emb_size
        batch_size = captions.size(0)
        seq_len = captions.size(1)
        enc_out = enc_out.view(batch_size, -1, self.enc_size) # batch_size * hw * 2048
        h = self.fc_h(enc_out.mean(dim=1))
        c = self.fc_c(enc_out.mean(dim=1))
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).cuda()
        for t in range(seq_len):
            context_vec, alpha = self.attention(enc_out, h)
            gate = F.sigmoid(self.fc_gate(h))
            context_vec = context_vec * gate
            h, c = self.lstm_cell(torch.cat((captions[:, t], context_vec), dim=1), (h, c))
            output_step = self.fc(h) # batch_size * vocab_size
            outputs[:, t] = output_step
        return outputs, (h, c)

    def greedy_pred(self, enc_out, pred_len):
        batch_size = enc_out.size(0)
        init = torch.ones((batch_size), dtype=int) *  2
        next_in = init.cuda()
        enc_out = enc_out.view(batch_size, -1, self.enc_size) # batch_size * hw * 2048
        h = self.fc_h(enc_out.mean(dim=1))
        c = self.fc_c(enc_out.mean(dim=1))
        outputs = torch.ones((batch_size, pred_len), dtype=int).cuda()
        for t in range(pred_len):
            embedding = self.embed(next_in) # batch_size * emb_size
            context_vec, alpha = self.attention(enc_out, h)
            gate = F.sigmoid(self.fc_gate(h))
            context_vec = context_vec * gate
            h, c = self.lstm_cell(torch.cat((embedding, context_vec), dim=1), (h, c))
            output_step = self.fc(h)
            next_in = output_step.argmax(dim=1)
            outputs[:, t] = next_in
        return outputs

    def beam_search_pred(self, enc_outs, pred_len, beam_size=3):
        batch_size = enc_outs.size(0)
        enc_outs = enc_outs.view(batch_size, -1, self.enc_size) # batch_size * hw * 2048
        outputs = torch.ones((batch_size, pred_len), dtype=int) # place_holder for outputs
        for idx in range(batch_size):
            enc_out = enc_outs[idx] # hw * 2048
            enc_out = enc_out.unsqueeze(0) # 1 * hw * enc_size
            enc_out = enc_out.repeat(beam_size, 1, 1) # beam_size * hw * enc_size, (view beam_size as batch_size for convenience)
            k_words = torch.ones((beam_size), dtype=int).cuda() * 2 # beam_size
            seqs = k_words.unsqueeze(0) # 1 * beam_size
            k_scores = torch.zeros(beam_size, 1).cuda() # beam_size * 1
            h = self.fc_h(enc_out.mean(dim=1)) # beam_size * enc_size
            c = self.fc_c(enc_out.mean(dim=1)) # beam_size * enc_size

            for step in range(pred_len):
                embedding = self.embed(k_words) # beam_size * emb_size
                context_vec, alpha = self.attention(enc_out, h)
                gate = F.sigmoid(self.fc_gate(h))
                context_vec = context_vec * gate
                h, c = self.lstm_cell(torch.cat((embedding, context_vec), dim=1), (h, c)) # beam_size * hidden_size
                scores = self.fc(h) # beam_size * vocab_size
                scores = F.log_softmax(scores, dim=1) # beam_size * vocab_size
                scores = k_scores.expand_as(scores) + scores # beam_size * vocab_size
                if step == 0:
                    k_scores, k_words = scores[0].topk(beam_size)
                else:
                    k_scores, k_words = scores.view(-1).topk(beam_size)
                prev_idx = k_words / self.vocab_size # beam_size
                next_idx = k_words % self.vocab_size # beam_size
                h = h[prev_idx]
                c = c[prev_idx]
                k_words = next_idx
                seqs = seqs[:, prev_idx] # L * beam_size
                seqs = torch.cat((seqs, next_idx.unsqueeze(0)), dim=0)
                k_scores = k_scores.unsqueeze(-1) # beam_size * 1
            output = seqs[:, k_scores.squeeze().argmax()]
            outputs[idx, :] = output[1:] # Don't include <sos>
        return outputs