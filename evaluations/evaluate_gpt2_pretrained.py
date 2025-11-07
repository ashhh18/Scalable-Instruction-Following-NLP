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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.add_special_tokens({"eos_token": "<|eos|>"})    
    return tokenizer

def evaluate_model(model, tokenizer, num_stories=10, num_repeats=2):
    with open("data/store/50_stories.json", "r") as f:
        stories = json.load(f)

    stories = stories[:10]
    avg_grammar = 0
    avg_creativity = 0
    avg_consistency = 0 
    avg_plot_sense = 0 
    avg_vocabulary_diversity = 0 
    ct = 0

    for story in tqdm(stories):
        ct += 1
        if (ct==10):
            break
        story = story.split()
        story = story[: random.randint(4, len(story) // 2)]
        story = " ".join(story)
        input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
        output = model.generate(
            input_ids,
            max_length=256,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        story_for_prompt = " ***" + output_text[len(story) :]

        eval_msg = gemini_eval(story_for_prompt)
        evals = json.loads(eval_msg)

        avg_grammar += evals["grammar"]
        avg_creativity += evals["creativity"]
        avg_consistency += evals["consistency"]
        avg_plot_sense += evals["plot_sense"]
        avg_vocabulary_diversity += evals["vocabulary_diversity"]

    avg_grammar /= 10 
    avg_creativity /= 10 
    avg_consistency /= 10 
    avg_plot_sense /= 10 
    avg_vocabulary_diversity /= 10

    return {"avg_grammar": avg_grammar,"avg_creativity": avg_creativity,"avg_consistency": avg_consistency,"avg_plot_sense": avg_plot_sense,"avg_vocabulary_diversity": avg_vocabulary_diversity}


def main():

    tokenizer = custom_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large").to(device)
    ret = evaluate_model(model, tokenizer)
    json.dump(ret, open("data/store-json/gpt2_pretrained_eval.json", "w"), indent=4)

if __name__ == "__main__":
    main()