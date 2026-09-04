import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model

from llm2vec import LLM2Vec


from tqdm import tqdm

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from torch.utils.data import Dataset
import json
import torch
import numpy as np
import pandas as pd

import pandas as pd
from torch.utils.data import Dataset
import json
import torch

class UnlearningDataset(Dataset):
    def __init__(self, args, retain_csv_path, forget_csv_path):
        self.retain_data = pd.read_csv(retain_csv_path)
        self.n_augment = args.n_augment
        
        self.forget_data = pd.read_csv(forget_csv_path)

        random_vector = torch.rand(1, 1, args.hidden_size)
        #control_vector = random_vector / torch.norm(random_vector) * args.steering_coeff
        control_vector = random_vector / (torch.norm(random_vector) + 1e-6) * args.steering_coeff

        self.control_vector = control_vector

    def __len__(self):
        return len(self.retain_data) + len(self.forget_data)

    def __getitem__(self, idx):
        if idx < len(self.retain_data):
            # 处理 retain 样本
            row = self.retain_data.iloc[idx]
            original_text = row[0]
            augmented_texts = [row[1:]]
            return {
                "type": "retain",
                "original": original_text,
                "augmented": augmented_texts
            }
        else:
            # 处理 forget 样本
            idx -= len(self.retain_data)
            text = self.forget_data[idx]
            return {
                "type": "forget",
                "text": text,
                "control_vector": self.control_vector
            }

@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        retain_batch = [item for item in batch if item["type"] == "retain"]
        forget_batch = [item for item in batch if item["type"] == "forget"]

        if len(retain_batch) == 0 or len(forget_batch) == 0:
            return None 

        # 处理 retain 样本（原始 + 增强）
        retain_texts = []
        for item in retain_batch:
            retain_texts.append(item["original"])  # 原始文本
            retain_texts.extend(item["augmented"])  # 增强文本

        # 处理 forget 样本
        forget_texts = [item["text"] for item in forget_batch]
        control_vector = forget_batch[0]["control_vector"] if len(forget_batch) > 0 else None

        retain_features = self.model.tokenize(retain_texts)
        forget_features = self.model.tokenize(forget_texts)

        return retain_features, forget_features, control_vector 
    
def l_con_retain(retain_embs, forget_embs, n_augment, temp=0.1):
    """
    retain_embs: (batch_retain * (1+n_augment), dim)
    forget_embs: (batch_forget, dim)
    """
    # 构造相似度矩阵
    sim_matrix = torch.mm(retain_embs, torch.cat([retain_embs, forget_embs], 0).T) / temp
    
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool).to(retain_embs.device)
    neg_mask = torch.zeros_like(sim_matrix, dtype=torch.bool).to(retain_embs.device)
    
#for循环掩码思路
    # for i in range(len(retain_embs)):
    #     if i%(n_augment+1)==0:
    #         candidates = [i+1+k for k in range(n_augment)]
    #         candidates += [k for k in range(len(retain_embs)) if k%(n_augment+1)==0 and k!=i]
    #         candidates += [k+len(retain_embs)+1 for k in range(len(forget_embs))]
    #         for j in candidates:
    #             pos_mask[i,j] = True

    base_indices = torch.arange(0, retain_embs.size(0), n_augment+1)   #给pivot元素
    aug_offsets = torch.arange(1, n_augment+1) 
    aug_indices = base_indices.unsqueeze(1) + aug_offsets.unsqueeze(0)
    pos_mask[base_indices.unsqueeze(1), aug_indices] = True
    pos_mask[base_indices.unsqueeze(1), base_indices.unsqueeze(0)] = True  # 各pivot样本之间positive pair
    pos_mask[base_indices, base_indices] = False  # 排除自身匹配

    exp_sim = torch.exp(sim_matrix)
    pos_sum = (exp_sim * pos_mask).sum(1)

    forget_indices = torch.arange(retain_embs.size(0), retain_embs.size(0) + forget_embs.size(0))
    neg_mask[base_indices.unsqueeze(1), forget_indices.unsqueeze(0)] = True
    # 计算对比损失
    neg_sum = (exp_sim * neg_mask).sum(1)
    
    return -torch.log(pos_sum / (pos_sum + neg_sum)).mean()

def l_con_forget(retain_embs, forget_embs, control_vectors, temp=0.1):
    """
    forget_embs: (batch_forget, dim)
    control_vectors: (batch_forget, dim) (所有向量相同)
    """
    targets = torch.cat([control_vectors, forget_embs, retain_embs], dim=0)
    
    # 计算相似度矩阵
    sim_matrix = torch.mm(forget_embs, targets.T) / temp
    
    # 生成样本掩码
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool).to(retain_embs.device)
    neg_mask = torch.zeros_like(sim_matrix, dtype=torch.bool).to(retain_embs.device)
    len_batch_forget = len(forget_embs)

    pos_mask[:, 0] = True
    # 所有样本对其他 forget 样本为正样本，排除自身
    pos_mask[:, 1:len_batch_forget + 1] = True
    pos_mask[:, 1:].fill_diagonal_(False)

#for循环控制逻辑    
    # for i in range(batch_forget):
    #     # Hard positive: 控制向量（第一个位置）
    #     pos_mask[i, 0] = True
    #     # Weak positives: 其他forget样本
    #     pos_mask[i, 1:batch_forget] = True
    #     pos_mask[i, i+1] = False  # 排除自己
    
    # 计算对比损失
    exp_sim = torch.exp(sim_matrix)
    pos_sum = (exp_sim * pos_mask).sum(1)

    neg_mask[:, len_batch_forget+1:] = True
    neg_sum = (exp_sim * neg_mask).sum(1)
    
    return -torch.log(pos_sum / (pos_sum + neg_sum)).mean()


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    retain_csv_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    forget_csv_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    simcse_dropout: float = field(
        default=0.1, metadata={"help": "The SimCSE dropout rate for the model"}
    )

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )
    
    forget_weight: float = field(default=0.5, metadata={"help": "The weight of forget loss"})

    n_augment: int = field(default=1, metadata={"help": "The number of augemented retain sample"})
    
    temp: float = field(default=0.1, metadata={"help": "The temperature of CL"})

    steering_coeff: float = field(default=300.0, metadata={"help": "steering_coeff of random fixed vector"})

    hidden_size: float = field(default=4096, metadata={"help": "size of embedding vector"})




class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class ContrastiveUnlearnTrainer(Trainer):
    def __init__(
        self,
        *args,
        forget_weight = 0.5,
        n_augment = 1,
        temp = 0.1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.forget_weight = forget_weight
        self.n_augment = n_augment
        self.temp = temp

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
        r_reps = self.model(inputs["retain_features"])
        f_reps = self.model(inputs["forget_features"])
        control_vector = inputs["control_vector"]

        loss = l_con_forget(r_reps, f_reps, self.n_augment, self.temp) + self.forget_weight*l_con_retain(r_reps, f_reps, control_vector, self.n_augment, self.temp)

        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()
    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    train_dataset = UnlearningDataset(custom_args, data_args.retain_csv_path, data_args.forget_csv_path)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        attention_dropout=custom_args.simcse_dropout,
    )

    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    tokenizer = model.tokenizer

    data_collator = DefaultCollator(model)

    trainer = ContrastiveUnlearnTrainer(
        model=model,
        args=training_args,
        forget_weight = custom_args.forget_weight,
        n_augment = custom_args.n_augment,
        temp = custom_args.temp,
        train_dataset= train_dataset,
        data_collator= data_collator,
        tokenizer= tokenizer
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    trainer.train()


if __name__ == "__main__":
    main()
