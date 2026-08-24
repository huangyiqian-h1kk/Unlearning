import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from torch.utils.data import Sampler

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

from llm22vec import LLM2Vec
import random

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

from torch.utils.data import Dataset, DataLoader
import json
import torch
import numpy as np
import pandas as pd

import pandas as pd
from torch.utils.data import Dataset
import json
import torch

class UnlearningDataset(Dataset):
    def __init__(self, args, retain_csv_path, forget_csv_path, batch_size,tokenizer):
        retain_data = pd.read_csv(retain_csv_path)
        forget_data = pd.read_csv(forget_csv_path)

        init_ratio = len(retain_data)/(len(retain_data) + len(forget_data))

        self.refined_retain_batch_size = round(init_ratio*batch_size)
        self.refined_forget_batch_size = round((1-init_ratio)*batch_size)

        self.batch_num = min(len(retain_data)//self.refined_retain_batch_size, len(forget_data)//self.refined_forget_batch_size)
        self.retain_data = retain_data.iloc[:self.refined_retain_batch_size*self.batch_num]
        self.forget_data = forget_data.iloc[:self.refined_forget_batch_size*self.batch_num]

        # self.retain_data = pd.read_csv(retain_csv_path)
        self.n_augment = args.n_augment
        
        # self.forget_data = pd.read_csv(forget_csv_path)

        random_tokens = random.sample(
            list(tokenizer.vocab.keys())[100:2000],  # 跳过特殊token
            k=random.randint(100, 500)
        )
        random_text = " ".join(random_tokens)
        self.random_text = random_text

    def __len__(self):
        return len(self.retain_data) + len(self.forget_data)

    def __getitem__(self, idx):
        if idx < len(self.retain_data):
            # 处理 retain 样本
            row = self.retain_data.iloc[idx]
            original_text = row[0]
            augmented_texts = list(row[1:])
            return {
                "type": "retain",
                "original": original_text,
                "augmented": augmented_texts
            }
        else:
            # 处理 forget 样本
            idx -= len(self.retain_data)
            text = self.forget_data.iloc[idx][0]
            return {
                "type": "forget",
                "text": text,
                "random_text": self.random_text
            }

class ProportionalBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.total_samples = len(dataset)
        
        # 生成索引映射表
        self.retain_indices = np.arange(len(self.dataset.retain_data))
        self.forget_indices = np.arange(len(self.dataset.retain_data), len(self.dataset.retain_data) + len(self.dataset.forget_data))

        if self.shuffle:
            np.random.shuffle(self.retain_indices)
            np.random.shuffle(self.forget_indices)

    def __iter__(self):
        retain_ptr = 0
        forget_ptr = 0
        
        for _ in range(self.dataset.batch_num):
            # 获取当前batch的索引
            retain_batch = self.retain_indices[
                retain_ptr : retain_ptr + self.dataset.refined_retain_batch_size
            ]
            forget_batch = self.forget_indices[
                forget_ptr : forget_ptr + self.dataset.refined_forget_batch_size
            ]

            # 合并索引并打乱顺序
            combined = np.concatenate([retain_batch, forget_batch])
            np.random.shuffle(combined)
            yield combined.tolist()
            
            # 移动指针
            retain_ptr += self.dataset.refined_retain_batch_size
            forget_ptr += self.dataset.refined_forget_batch_size

    def __len__(self):
        return self.dataset.batch_num

@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        retain_batch = [item for item in batch if item["type"] == "retain"]
        forget_batch = [item for item in batch if item["type"] == "forget"]
        if len(retain_batch) == 0 or len(forget_batch) == 0:
            return None, None, None
        # 处理 retain 样本（原始 + 增强）
        retain_texts = []
        for item in retain_batch:
            retain_texts.append(item["original"])  # 原始文本
            retain_texts.extend(item["augmented"])  # 增强文本

        # 处理 forget 样本
        forget_texts = [item["text"] for item in forget_batch]
        random_text = forget_batch[0]["random_text"] if len(forget_batch) > 0 else None
        
        retain_features = self.model.tokenize(retain_texts)
        forget_features = self.model.tokenize(forget_texts)
        random_features = self.model.tokenize([random_text])
        return retain_features, forget_features, random_features

def l_con_retain_l2(retain_embs, forget_embs, n_augment, temp=0.1):
    """
    retain_embs: (batch_retain * (1+n_augment), dim)
    forget_embs: (batch_forget, dim)
    """
    # 构造目标向量
    targets = torch.cat([retain_embs, forget_embs], 0)
    
    # 提取pivot和augment样本
    row_indices = torch.arange(retain_embs.shape[0])
    pivot_rows = row_indices[row_indices % (1+n_augment) == 0]
    retain_pivots = retain_embs[pivot_rows]

    # 计算平方欧氏距离矩阵
    a = retain_pivots
    b = targets
    a_sq = (a ** 2).sum(dim=1, keepdim=True)  # (m, 1)
    b_sq = (b ** 2).sum(dim=1)               # (n,)
    ab = torch.mm(a, b.t())                  # (m, n)
    dists = a_sq + b_sq.unsqueeze(0) - 2 * ab
    sim_matrix = -dists / temp  # 将距离转换为相似度

    print(l_con_retain_l2)
    # 构造正负样本掩码（保持与原始代码相同）
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool).to(retain_embs.device)
    neg_mask = torch.zeros_like(sim_matrix, dtype=torch.bool).to(retain_embs.device)
    
    # 使用矢量化方式构造掩码
    i = torch.arange(pos_mask.shape[0]).view(-1, 1)
    j = torch.arange(pos_mask.shape[1]).view(1, -1)
    mask1 = (j > i * (n_augment+1)) & (j < (i+1) * (n_augment+1)) & (j < retain_embs.shape[0])
    mask2 = (j % (n_augment+1) == 0) & (j != i * (n_augment+1)) & (j < retain_embs.shape[0])
    pos_mask[mask1 | mask2] = True
    
    # 构造负样本掩码
    k = torch.arange(neg_mask.size(1))
    forget_indices = k >= retain_embs.shape[0]
    neg_mask[:, forget_indices] = True

    # 计算对比损失
    exp_sim = torch.exp(sim_matrix)
    pos_sum = (exp_sim * pos_mask).sum(1)
    neg_sum = (exp_sim * neg_mask).sum(1)
    print("pos_mask")
    print(pos_mask)
    print("neg_mask")
    print(neg_mask)
    print("sim_matrix")
    print(sim_matrix)
    print("pos_sum")
    print(pos_sum)
    print("neg_sum")
    print(neg_sum)
    return -torch.log(pos_sum / (pos_sum + neg_sum)).mean()

def l_con_forget_l2(retain_embs, forget_embs, control_vectors, temp=10):
    """
    forget_embs: (batch_forget, dim)
    control_vectors: (batch_forget, dim) (所有向量相同)
    """
    # 构造目标向量
    targets = torch.cat([control_vectors, forget_embs, retain_embs], dim=0)
    
    # 计算平方欧氏距离矩阵
    a = forget_embs
    b = targets
    a_sq = (a ** 2).sum(dim=1, keepdim=True)  # (m, 1)
    b_sq = (b ** 2).sum(dim=1)               # (n,)
    ab = torch.mm(a, b.t())                  # (m, n)
    dists = a_sq + b_sq.unsqueeze(0) - 2 * ab
    sim_matrix = -dists / temp  # 将距离转换为相似度

    # 构造正负样本掩码（保持与原始代码相同）
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool).to(retain_embs.device)
    neg_mask = torch.zeros_like(sim_matrix, dtype=torch.bool).to(retain_embs.device)
    len_batch_forget = len(forget_embs)
    
    # 正样本掩码（控制向量和其他forget样本）
    pos_mask[:, 0] = True
    pos_mask[:, 1:len_batch_forget + 1] = True
    pos_mask[:, 1:].fill_diagonal_(False)  # 排除自己
    
    # 负样本掩码（retain样本）
    neg_mask[:, len_batch_forget+1:] = True

    # 计算对比损失
    exp_sim = torch.exp(sim_matrix)
    pos_sum = (exp_sim * pos_mask).sum(1)
    neg_sum = (exp_sim * neg_mask).sum(1)
    print("sim_matrix")
    print(sim_matrix)
    print("pos_sum")
    print(pos_sum)
    print("neg_sum")
    print(neg_sum)
    return -torch.log(pos_sum / (pos_sum + neg_sum)).mean()
    
def l_con_retain(retain_embs, forget_embs, n_augment, temp=0.1):
    """
    retain_embs: (batch_retain * (1+n_augment), dim)
    forget_embs: (batch_forget, dim)
    """
    # 构造相似度矩阵
    retain_embs = torch.nn.functional.normalize(retain_embs, p=2, dim=1)
    forget_embs = torch.nn.functional.normalize(forget_embs, p=2, dim=1)
    row_indices = torch.arange(retain_embs.shape[0])
    pivot_rows = row_indices[row_indices % (1+n_augment) == 0]
    augment_rows = row_indices[row_indices % (1+n_augment) != 0]
    retain_pivots = retain_embs[pivot_rows]
    retain_augments = retain_embs[augment_rows]


    sim_matrix = torch.mm(retain_pivots, torch.cat([retain_embs, forget_embs], 0).T) / temp
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

    i = torch.arange(pos_mask.shape[0]).view(-1, 1)  # 行索引
    j = torch.arange(pos_mask.shape[1]).view(1, -1)  # 列索引  
    # 计算布尔掩码
    mask1 = (j > i * (n_augment+1)) & (j < (i+1) * (n_augment+1)) & j < retain_embs.shape[0]
    mask2 = (j % (n_augment+1) == 0) & (j != i * (n_augment+1)) & j < retain_embs.shape[0]
    pos_mask[mask1 | mask2] = True
    exp_sim = torch.exp(sim_matrix)
    pos_sum = (exp_sim * pos_mask).sum(1)
   
    k = torch.arange(neg_mask.size(1)) 
    forget_indices = k >= 9  
    neg_mask[:, forget_indices] = True

    # 计算对比损失
    neg_sum = (exp_sim * neg_mask).sum(1)
    return -torch.log(pos_sum / (pos_sum + neg_sum)).mean()

def l_con_forget(retain_embs, forget_embs, control_vectors, temp=10):
    """
    forget_embs: (batch_forget, dim)
    control_vectors: (batch_forget, dim) (所有向量相同)
    """
    forget_embs = torch.nn.functional.normalize(forget_embs, p=2, dim=1)
    targets = torch.cat([control_vectors, forget_embs, retain_embs], dim=0)
    targets = torch.nn.functional.normalize(targets, p=2, dim=1)
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
    #print(f"Model's Lora trainable parameters:")
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
        forget_weight = 0.2,
        train_dataloader=None, 
        n_augment = 1,
        temp = 0.5,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.forget_weight = forget_weight
        self.n_augment = n_augment
        self.temp = temp
        self.train_dataloader = train_dataloader  

    def get_train_dataloader(self):
        if self.train_dataloader is not None:
            return self.train_dataloader
        else:
            return super().get_train_dataloader()      

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        r_reps = self.model(inputs[0])
       # r_reps =  torch.nn.functional.normalize(r_reps, p=2, dim=1)
        f_reps = self.model(inputs[1])
       # f_reps =  torch.nn.functional.normalize(f_reps, p=2, dim=1)
        control_vector = self.model(inputs[2])
        
        l_con_f = l_con_forget_l2(r_reps, f_reps, control_vector, self.temp) 
        l_con_r = l_con_retain_l2(r_reps, f_reps, self.n_augment, self.temp)
        contrastive_loss = self.forget_weight*l_con_f + (1-self.forget_weight)*l_con_r
        
        retain_features, forget_features, random_features = inputs
        forget_lm_outputs = model.model.model(input_ids=forget_features["input_ids"],attention_mask=forget_features["attention_mask"], labels=forget_features["input_ids"])
        retain_lm_outputs = model.model.model(input_ids=retain_features["input_ids"],attention_mask=retain_features["attention_mask"], labels=retain_features["input_ids"])

        lm_weight = 0.0001
        gamma = 0.5
        forget_lm_loss = forget_lm_outputs.loss
        retain_lm_loss = retain_lm_outputs.loss
        lm_loss = forget_lm_loss - gamma * retain_lm_loss
        total_loss = (1-lm_weight) * contrastive_loss - lm_weight * lm_loss

        if random.random()<0.99:
            print("l_con_f",l_con_f)
            print("l_con_r",l_con_r)
            print("contrastive_loss",contrastive_loss)
            print("lm_loss",lm_loss)
            print("total_loss",total_loss)
        return total_loss

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

    train_dataset = UnlearningDataset(custom_args, data_args.retain_csv_path, data_args.forget_csv_path, training_args.per_device_train_batch_size, tokenizer)

    sampler = ProportionalBatchSampler(
        train_dataset, 
        shuffle=True  # 是否打乱顺序
    )

    

    data_collator = DefaultCollator(model)

        # 初始化 DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=data_collator,  # 使用自定义的 collate_fn
        num_workers=4,  # 数据加载的线程数
        pin_memory=True  # 是否将数据加载到固定内存（加速 GPU 传输）
    )

    trainer = ContrastiveUnlearnTrainer(
        model=model,
        args=training_args,
        forget_weight = custom_args.forget_weight,
        n_augment = custom_args.n_augment,
        temp = custom_args.temp,
        train_dataloader=dataloader,
        tokenizer= tokenizer
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    trainer.train()


if __name__ == "__main__":
    main()

