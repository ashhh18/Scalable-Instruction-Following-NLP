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

def custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.add_special_tokens({"eos_token": "<|eos|>"})    
    return tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer):
    with open("data/store-json/reasoning_prompts.json", "r") as f:
        prompts = json.load(f)

    score = 0
    ct = 0
    for story in tqdm(prompts):
        story = story.split()
        story = story[: random.randint(1, len(story) // 2)]
        story = " ".join(story)

        input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
        output = model.model.generate(
            input_ids,
            max_length=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        story_for_prompt = " *** " + output_text[len(story) :]
        
        eval_msg = gemini_prompt(story_for_prompt,1)
        with open("data/krct-results/reasoning_stories.txt", "a") as file:
            file.write(story_for_prompt)
            file.write("\n\n\n")
        time.sleep(10)
        evals = json.loads(eval_msg)

        score += int(evals["reasoning ability"])

    score = score/ 9

    return {
        "reasoning_ability": score,
    }


def main():
    tokenizer = custom_tokenizer()
    if eval_file == "null":
        print ("eval file not provided\n")
        return
    with open("data/krct-results/reasoning_stories.txt", "a") as file:
        file.write(eval_file)
        file.write("\n")
    model = ScratchModel.load_from_checkpoint(
        eval_file, tokenizer=tokenizer
    ).to(device)
    
    ret = evaluate_model(model, tokenizer)
    print(ret)
    with open("eval2.txt", "a") as file: 
        file.write(f"{eval_file}\n")
        file.write(f"{ret}\n") 


if __name__ == "__main__":
    main()
