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
from config import eval_file, embed, layer_inp, heads_inp
from run_model import ScratchModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemini_api import gemini_eval

device = torch.device("cuda")

def custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.add_special_tokens({"eos_token": "<|eos|>"})    
    return tokenizer

model = ScratchModel.load_from_checkpoint(eval_file, tokenizer=custom_tokenizer()).to(device)
tokenizer = custom_tokenizer()

model.to("cuda")
model.eval()

prompt = "Once upon a time there was"

generated_stories = []

for i in range(50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=200, do_sample=True)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_stories.append(output_text)

output_file = "generated_stories.json"
with open(output_file, "w") as f:
    json.dump(generated_stories, f, indent=4)

print(f"50 generated stories saved to {output_file}")
