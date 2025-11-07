import os
import torch
import time
import sys
import json
import pandas
import numpy
import random
import torch
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import wandb
from tqdm import tqdm  
from config import eval_file
from scratchmodel import ScratchModel

NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_TOKENS = 10
LR = 1e-5
KL_FACTOR = 6000
REF_TEXT = "cat"
device = torch.device("cuda")

def custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.add_special_tokens({"eos_token": "<|eos|>"})    
    return tokenizer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
reference_embedding = embedding_model.encode(REF_TEXT, convert_to_tensor=True)

for param in embedding_model.parameters():
    param.requires_grad = False


def compute_rewards(sequences):
    sequence_embeddings = embedding_model.encode(sequences, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(
        reference_embedding.unsqueeze(0), sequence_embeddings
    ).squeeze()
    return cosine_similarities


tokenizer = custom_tokenizer()
if eval_file == "null":
    print ("eval file not provided\n")
    exit(0)

model = ScratchModel.load_from_checkpoint(eval_file, tokenizer=tokenizer).to(device)
ref_model = ScratchModel.load_from_checkpoint(eval_file, tokenizer=tokenizer).to(device)

optimizer = AdamW(model.parameters(), lr=LR)

for param in ref_model.parameters():
    param.requires_grad = False

prompt = "Once upon a time there was"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs", ncols=100):
    model.train()

    output_ids = torch.full(
        (BATCH_SIZE, NUM_TOKENS), tokenizer.eos_token_id, device="cuda"
    )
    output_ids[:, : input_ids.shape[1]] = input_ids

    log_probs_accumulated = torch.zeros((BATCH_SIZE, 1), device="cuda")
    kl_div_accumulated = torch.zeros((BATCH_SIZE, 1), device="cuda")

    active_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device="cuda")

    for i in tqdm(range(input_ids.shape[1], NUM_TOKENS), desc="Tokens", ncols=100):
        prompt = output_ids[:, :i].clone()
        logits = model(prompt).logits[:, -1, :]
        logits_active = logits[active_mask]
        if logits_active.shape[0] == 0:
            break
        probs = torch.nn.functional.softmax(logits_active, dim=-1)
        dist = Categorical(probs)
        next_tokens = dist.sample()
        log_probs_accumulated[active_mask] += dist.log_prob(next_tokens).unsqueeze(-1)
        output_ids[active_mask, i] = next_tokens

        ref_logits = ref_model(prompt).logits[:, -1, :]
        ref_logits_active = ref_logits[active_mask]

        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_active, dim=-1),
            torch.nn.functional.log_softmax(ref_logits_active, dim=-1),
            reduction="none",
            log_target=True,
        )
        kl_div_accumulated[active_mask] += kl_div.mean(dim=-1).unsqueeze(-1)

        finished = next_tokens == tokenizer.eos_token_id
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        new_mask = active_mask.clone()
        new_mask[active_indices] = ~finished
        active_mask = new_mask

    normalized_log_probs = log_probs_accumulated / NUM_TOKENS
    normalized_kl_div = kl_div_accumulated / NUM_TOKENS

    with torch.no_grad():
        sequences = [
            tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in output_ids
        ]
        rewards = compute_rewards(sequences)

    neg_advantage = (-normalized_log_probs * rewards.unsqueeze(-1)).mean()
    loss = neg_advantage + KL_FACTOR * normalized_kl_div.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if WANDB:
        wandb.log({"loss": loss, "reward": rewards.mean(), "kl": normalized_kl_div.mean()})

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}: Loss: {loss.item()} Rewards: {rewards.mean()} NegAdv: {neg_advantage} KL: {normalized_kl_div.mean()}"
    )

save_directory = "./checkpoints"
model.save_pretrained(save_directory)
