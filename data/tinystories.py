import os
import torch
import time
import sys
import json
import pandas
import numpy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from typing import Optional
from config import batch_size, max_length, cache_dir
 

class TextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.inputs = tokenized_data

    def __len__(self):
        return len(self.inputs)
 
    def __getitem__(self, idx):
        return self.inputs[idx]
 
 
class TinyStories:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.val_dataset = None
 
    def init_cache(self, stage: Optional[str] = None):
        cache_path_train = os.path.join(cache_dir, "syn_train.json")
        cache_path_val = os.path.join(cache_dir, "syn_val.json")
 
        print(cache_path_train)
        print(cache_path_val)
 
        if os.path.exists(cache_path_train) and os.path.exists(cache_path_val):
            with open(cache_path_train, "r") as f_train:
                self.train_dataset = TextDataset(json.load(f_train))
            with open(cache_path_val, "r") as f_val:
                self.val_dataset = TextDataset(json.load(f_val))

        else:
            temp_dataset = load_dataset("ashh18/synonymTiny")
            dataset = temp_dataset.shuffle(seed=42)
 
            tokenized_dataset = dataset.map(lambda x: self.tokenizer([item + self.tokenizer.eos_token for item in x["text"]],truncation=False,padding=False,),batched=True,num_proc=20,)

            train_chunks = self.split_data(tokenized_dataset["train"])
            val_chunks = self.split_data(tokenized_dataset["validation"])
 
            self.train_dataset = TextDataset(train_chunks)
            self.val_dataset = TextDataset(val_chunks)
 
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path_train, "w") as f_train:
                json.dump(train_chunks, f_train)
            with open(cache_path_val, "w") as f_val:
                json.dump(val_chunks, f_val)
 
    def split_data(self, tokenized_data):
        chunks = []
        for item in tqdm(tokenized_data):
            input_ids = item["input_ids"]
            for i in range(0, len(input_ids), max_length):
                chunk = input_ids[i : i + max_length]
                chunks.append(chunk)
        return chunks
 
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=batch_size,num_workers=20,shuffle=True,collate_fn=self.collate_fn,)
 
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=batch_size,num_workers=20,collate_fn=self.collate_fn,)
 
    def collate_fn(self, batch):
        
        padded_batch = pad_sequence([torch.LongTensor(chunk) for chunk in batch],batch_first=True,padding_value=self.tokenizer.pad_token_id,)
        attention_mask = (padded_batch != self.tokenizer.pad_token_id).int() 
        labels = padded_batch.clone() 
        return {"input_ids": padded_batch, "labels": labels, "attention_mask": attention_mask,}
 