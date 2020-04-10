# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:10:03 2020

@author: Zhe Cao
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchtext.data import get_tokenizer, Field

class Flickr30kDataset(Dataset):
    def __init__(self, data_path, csv_path, transforms, comment_num):
        self.data_path = data_path
        self.csv_path = csv_path
        self.data_files = os.listdir(self.data_path)
        self.data_files = sorted(self.data_files)
        self.transforms = transforms
        assert 0 <= comment_num <= 4
        self.comment_num = ' ' + str(comment_num)
        self._load_csv()
        self._tokenizer()
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.data_files[idx])
        img = Image.open(img_path)
        inputs = self.transforms(img)
        captions = self.token_list[idx]
        cap_lens = self.len_list[idx]
        return inputs, captions, cap_lens
    
    def __len__(self):
        return len(self.data_files)
    
    def _load_csv(self):
        self.results = pd.read_csv(self.csv_path, sep='|')
        fix_19999 = self.results.loc[19999][' comment_number']
        self.results.loc[19999][' comment_number'] = ' 4'
        self.results.loc[19999][' comment'] = fix_19999[4:]
        self.results = self.results.sort_values(by=[' comment_number', 'image_name', ])
    
    def _tokenizer(self):
        self.caption_list = self.results.loc[self.results[' comment_number'] == self.comment_num][' comment'].tolist()
        tokenizer = get_tokenizer("basic_english")
        self.token_list = [tokenizer(caption) for caption in self.caption_list]
        self.len_list = torch.tensor([len(token) for token in self.token_list])
        self.len_list += 1 # allow for <sos> or <eos>
        self.seq_len = self.len_list.max() + 1
        self.field = Field(tokenize='spacy', tokenizer_language='en', 
                           init_token='<sos>', eos_token='<eos>', lower=True, fix_length=self.seq_len)
        self.field.build_vocab(self.token_list)
        self.token_list = self.field.process(self.token_list)
        self.token_list = self.token_list.transpose(1, 0)
        
    def get_reference_corpus(self):
        reference_corpus = []
        for num in range(5):
            caption_candidates = self.results.loc[self.results[' comment_number'] == ' '+str(num)][' comment'].tolist()
            reference_corpus.append([x.split() for x in caption_candidates])
        reference_corpus = np.array(reference_corpus)
        reference_corpus = reference_corpus.transpose(1, 0)
        return reference_corpus
    
    