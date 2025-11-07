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
from gemini_api import gemini_eval,gemini_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.add_special_tokens({"eos_token": "<|eos|>"})    
    return tokenizer
    
def evaluate_model(model, tokenizer):
    with open("data/store-json/instruct.json", "r") as f:
        stories = json.load(f)
    
    stories = stories[:10]
    ct = 0
    for orig_story in tqdm(stories):
        ct += 1
        if ct == 10:
            break
        orig_story = "Words: Cat\nStory: "
        st = "Given are the features, words or the summary of a story. You should write the story so that it aligns with them."
        ll = len(st)
        story = st + orig_story
        story = story
        print("story:", story)
        input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
        output = model.model.generate(
            input_ids,
            max_length=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        story_for_prompt = story[ll:] + output_text[len(story) :]

        print("story_for_prompt:", story_for_prompt)
    
    
    return 0
    
def main():
    tokenizer = custom_tokenizer()
    model = ScratchModel.load_from_checkpoint(
        eval_file, tokenizer=tokenizer
    ).to(device)
    
    ret = evaluate_model(model, tokenizer)
    
    
if __name__ == "__main__":
    main()