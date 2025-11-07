import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import pandas as pd
import json
from scratchmodel import ScratchModel 
from config import eval_file, embed, layer_inp, heads_inp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.add_special_tokens({"eos_token": "<|eos|>"})    
    return tokenizer
 
class AttentionHeads:
    def __init__(self, model_path):
        self.tokenizer = custom_tokenizer()
        self.model = ScratchModel.load_from_checkpoint(model_path,tokenizer=self.tokenizer)
        self.model = self.model.to(device)
        self.model.eval()
        
    def get_type(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = tokens.to(device)
        with torch.no_grad():
            output = self.model.model(tokens.input_ids, attention_mask=tokens.attention_mask, output_attentions=True)
        
        attention_head_analysis = output.attentions[-1].squeeze(0)
        return attention_head_analysis, tokens.input_ids[0]
 
    def heat_map(self, item, attention_pattern, tokens, save_path=None):
        attention_pattern = attention_pattern.cpu()
        att_map = attention_pattern[item].numpy()
        token_labels = [self.tokenizer.decode(t) for t in tokens]
        
        plt.figure(figsize=(22, 18))
        sns.heatmap(att_map, xticklabels=token_labels, yticklabels=token_labels,cmap="coolwarm")
        plt.title(f'Attention Head {item}')
        if save_path:
            plt.savefig(save_path)
        plt.close()
 
 
def main():
    interpreter = AttentionHeads(eval_file)
    text = text = """One day, Lucy asked Tom, "I am looking for a banana, but I can’t find it."\n\nTom said, "Don’t worry, I will help you."\n\nLucy and Tom went to the park. They looked for the banana together. After a while, they found the banana.\n\nLucy was happy. She said, "Thank you, Tom. You are a good friend."\n\nTom replied, "You are welcome, Lucy. I am happy to help you. Let’s eat the banana together!" """
    attention_head_analysis, tokens = interpreter.get_type(text)
    
    for item in range(attention_head_analysis.shape[0]):
        interpreter.heat_map(
            item, 
            attention_head_analysis, 
            tokens,
            f"data/attn-heatmap/attention_head_{item}_{embed}_{layer_inp}_{heads_inp}.png"
        )

 
if __name__ == "__main__":
    main()