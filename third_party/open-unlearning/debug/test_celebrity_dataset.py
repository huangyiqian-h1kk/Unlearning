#!/usr/bin/env python3
"""测试celebrity数据集的loss mask是否正确"""

import torch
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.celebrity_unlearn import CelebrityUnlearnDataset
from data.collators import DataCollatorForSupervisedDataset
from data.utils import IGNORE_INDEX

def test_loss_mask():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试样本
    test_samples = [
        {"0": "the age at death of Nelson Mandela is 95"},  # statement
        {"0": "Question: what is the birth year of Nelson Mandela? Answer: the birth year of Nelson Mandela is 1918"},  # QA
        {"0": "Nelson Mandela was (?) years old when he passed away. (?) : 95"}  # paraphrased
    ]
    
    # 模拟数据集处理
    processed_samples = []
    for i, data in enumerate(test_samples):
        text = data["0"]
        
        # 创建临时数据集实例来测试处理逻辑
        dataset = CelebrityUnlearnDataset(
            hf_args={"path": "csv", "data_files": "dummy.csv", "split": "train"},
            template_args={},
            tokenizer=tokenizer,
        )
        
        # 手动处理样本
        if dataset._is_qa(text):
            result = dataset._process_qa(text, i)
            sample_type = "QA"
        elif dataset._is_paraphrased(text):
            result = dataset._process_paraphrased(text, i)
            sample_type = "Paraphrased"
        else:
            result = dataset._process_statement(text, i)
            sample_type = "Statement"
        
        result["sample_type"] = sample_type
        processed_samples.append(result)
    
    # 测试DataCollator
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="left")
    
    print("=== Individual Samples ===")
    for i, sample in enumerate(processed_samples):
        print(f"\nSample {i} ({sample['sample_type']}):")
        print(f"Text: {test_samples[i]['0'][:80]}...")
        
        # 显示哪些位置会计算loss
        labels = sample['labels']
        loss_positions = (labels != IGNORE_INDEX).sum().item()
        total_positions = len(labels)
        print(f"Loss positions: {loss_positions}/{total_positions}")
        
        # 显示前10个token的loss计算情况
        input_ids = sample['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[:15])
        labels_slice = labels[:15]
        print("First 15 tokens loss status:")
        for j, (token, label) in enumerate(zip(tokens, labels_slice)):
            status = "LOSS" if label != IGNORE_INDEX else "NO_LOSS"
            print(f"  {j:2d}: {token:15s}: {status}")
    
    # 测试批处理
    print("\n=== After DataCollator ===")
    batch = collator(processed_samples)
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    
    # 验证每个样本的loss计算
    for i in range(len(processed_samples)):
        sample_labels = batch['labels'][i]
        ignore_count = (sample_labels == IGNORE_INDEX).sum().item()
        loss_count = ((sample_labels != IGNORE_INDEX) & 
                     (sample_labels != tokenizer.pad_token_id)).sum().item()
        print(f"Sample {i}: {ignore_count} ignored, {loss_count} loss positions")

if __name__ == "__main__":
    test_loss_mask()
