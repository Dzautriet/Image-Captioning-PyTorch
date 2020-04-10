# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:59:12 2020

@author: Zhe Cao
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchtext
import os
import gc
from torchvision import models
from load_data import Flickr30kDataset
from transformer import Encoder, Decoder
from utils import AverageMeter, mask_accuracy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

data_path = 'flickr30k_images/flickr30k_images'
csv_path = 'flickr30k_images/results.csv'
save_path = "."
use_pretrained_embed = False

epochs = 100
batch_size = 32
enc_lr = 1e-4
dec_lr = 1e-4
patience = 10
enc_save_path, dec_save_path = os.path.join(save_path, 'best_enc_trans'), os.path.join(save_path, 'best_dec_trans')
best_acc = 0
best_epoch = 0
channels = 2048
emb_size = 300
nHead = 10
hidden_size = 300
nLayers = 3
dropout_dec = 0.2
dropout_pos = 0.1

if __name__ == "__main__":
    # Instantiate data set
    transforms_train = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_set = Flickr30kDataset(data_path, csv_path, transforms_train, 3)
    reference_corpus = data_set.get_reference_corpus()
    vocab_size = len(data_set.field.vocab.itos)
    
    # Split data set
    num_data = len(data_set)
    idx = list(range(num_data))
    train_set = Subset(data_set, idx[:-2000])
    vali_set = Subset(data_set, idx[-2000:-1000])
    test_set = Subset(data_set, idx[-1000:])
    
    # Pretrained word embedding
    if use_pretrained_embed:
        w2v = torchtext.vocab.GloVe(name = "840B", dim = 300)
        vocab_size = len(data_set.field.vocab.itos)
        embedding_matrix = torch.zeros(vocab_size, 300)
        for idx, word in enumerate(data_set.field.vocab.itos):
            embedding_matrix[idx] = w2v.get_vecs_by_tokens(word)
    else:
        embedding_matrix = None
    
    # Data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    vali_loader = DataLoader(vali_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    # Instantiate models
    resnet = models.resnet50(pretrained=True)
    encoder = Encoder(resnet, channels, emb_size)
    encoder.freeze_all()
    decoder = Decoder(vocab_size, emb_size, nHead, hidden_size, nLayers, dropout_dec, dropout_pos, use_pretrained_embed, embedding_matrix)
    decoder, encoder = decoder.cuda(), encoder.cuda()
    del resnet
    gc.collect()
    torch.cuda.empty_cache()
    
    # Criterion and optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=data_set.field.vocab.stoi['<pad>'])
    enc_optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=enc_lr)
    dec_optimizer = torch.optim.AdamW(decoder.parameters(), lr=dec_lr)
    
    # Train model
    train_losses = []
    vali_losses = []
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        train_loss = AverageMeter()
        vali_loss = AverageMeter()
        vali_acc = AverageMeter()
        for batch_index, (inputs, captions, caplens) in enumerate(train_loader):
            inputs, captions = inputs.cuda(), captions.cuda()
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            enc_out = encoder(inputs)
            captions_input = captions[:, :-1]
            captions_target = captions[:, 1:]
            output = decoder(captions_input, enc_out)
            output = output.permute(1, 0, 2)
            output_shape = output.reshape(-1, vocab_size)
            loss = criterion(output_shape, captions_target.reshape(-1))
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            train_loss.update(loss.item(), inputs.size(0))
            if batch_index % 40 == 0:
                print("Batch: {}, loss {:.4f}.".format(batch_index, loss.item()))
        # Evaluation
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for batch_index, (inputs, captions, caplens) in enumerate(vali_loader):
                inputs, captions = inputs.cuda(), captions.cuda()
                enc_out = encoder(inputs)
                captions_input = captions[:, :-1]
                captions_target = captions[:, 1:]
                output = decoder(captions_input, enc_out)
                output = output.permute(1, 0, 2)
                output_shape = output.reshape(-1, vocab_size)
                loss = criterion(output_shape, captions_target.reshape(-1))
                acc = mask_accuracy(output_shape, captions_target.reshape(-1), ignore_index=data_set.field.vocab.stoi['<pad>'])
                vali_loss.update(loss.item(), inputs.size(0))
                vali_acc.update(acc, inputs.size(0))
        print("Epoch: {}/{}, training loss: {:.4f}, vali loss: {:.4f}, vali acc\: {:.4f}.\
              ".format(epoch, epochs, train_loss.avg, vali_loss.avg, vali_acc.avg))
        train_losses.append(train_loss.avg)
        vali_losses.append(vali_loss.avg)
        # Save best
        if vali_acc.avg > best_acc:
            best_acc = vali_acc.avg
            best_epoch = epoch
            torch.save(encoder.state_dict(), enc_save_path)
            torch.save(decoder.state_dict(), dec_save_path)
        # Early stopping
        if epoch - best_epoch >= patience:
            print("Early stopping")
            break
    