import os
import torch
import time
import sys
import json
import pandas
import numpy
import random
import pytorch_lightning
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from config import  max_length, model_config


class ScratchModel (pytorch_lightning.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        gpt2_config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=max_length,
            n_ctx=max_length,
            n_embd=model_config["hidden_size"],
            n_layer=model_config["layers"],
            n_head=model_config["heads"],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        self.model = GPT2LMHeadModel(gpt2_config)
        self.config = gpt2_config

        self.learning_rate = 0.0005
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def text_gen(self, prompt, max_length=50, num_return_sequences=1):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_return_sequences=num_return_sequences,
            attention_mask=(input_ids != self.pad_token_id).int(),
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]

    def trainer(self, input_ids, attention_mask, labels):
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def validater(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def tester(self, input_ids):
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.pad_token_id,
                num_return_sequences=1,
            )
        return [self.tokenizer.decode(output.cpu().numpy(), skip_special_tokens=True) for output in output_ids]

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
