from typing import Optional
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

class LLM2Vec2CausalLM(nn.Module):
    def __init__(self, llm2vec_model):
        super().__init__()
        self.llm = llm2vec_model
        self.inner_model = llm2vec_model.model  # huggingface + peft
        self.tokenizer = llm2vec_model.tokenizer
        self.config = self.inner_model.config

    def generate(self, *args, **kwargs):
        return self.inner_model.generate(*args, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        output = self.inner_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return CausalLMOutputWithPast(
            loss=output.loss,
            logits=output.logits,
        )

