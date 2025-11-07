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
from scratchmodel import ScratchModel
from data.tinystories import TinyStories, TextDataset
from config import model_name

device = torch.device("cuda")

def custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.add_special_tokens({"eos_token": "<|eos|>"})    
    return tokenizer

epochs = 1

os.makedirs("/scratch/ameyar3103/", exist_ok=True)

def train(model, data_module: TinyStories, device):
    optimizer = model.optimizer()
    model.to(device)

    data_module.init_cache()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    print(len(train_loader))

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Progress")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model.trainer(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Training Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation Progress")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss = model.validater(input_ids, attention_mask, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

        checkpoint_path = os.path.join("/scratch/ameyar3103/", f"{model_name}_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

def main():
    tokenizer = custom_tokenizer()

    model = ScratchModel(tokenizer)
    data_module = TinyStories(tokenizer)

    train(model, data_module, device)

if __name__ == "__main__":
    main()
