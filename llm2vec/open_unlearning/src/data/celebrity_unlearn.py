import torch
import re
from torch.utils.data import Dataset
from data.utils import load_hf_dataset, add_dataset_index, IGNORE_INDEX

class CelebrityUnlearnDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        text_key="0",
        max_length=512,
        predict_with_generate=False
    ):
        super(CelebrityUnlearnDataset, self).__init__()
        self.tokenizer = tokenizer
        self.template_args = template_args
        self.max_length = max_length
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        
        # Load dataset
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)

    def _is_statement(self, text):
        """判断是否为statement格式: "the X of Y is Z" """
        return text.startswith("the ") and " of " in text and " is " in text

    def _is_qa(self, text):
        """判断是否为QA格式"""
        return text.startswith("Question:") and "Answer:" in text

    def _is_paraphrased(self, text):
        """判断是否为重新表述的格式（包含(?) :）"""
        return "(?)" in text and "(?) :" in text

    def _process_statement(self, text, index):
        """处理statement格式 - 对整个文本计算loss（除第一个token）"""
        tokens = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        # 对整个序列计算loss（除了第一个token）
        labels = input_ids.clone()
        if len(labels) > 1:
            labels[0] = IGNORE_INDEX  # 不对第一个token计算loss
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "index": index,
            "text_type": "statement"
        }

    def _process_qa(self, text, index):
        """处理QA格式 - 只对Answer部分计算loss"""
        if "Answer:" not in text:
            return self._process_statement(text, index)
        
        # 分割问题和答案
        parts = text.split("Answer:", 1)
        question_part = parts[0].strip()
        answer_part = parts[1].strip()
        
        # 构建完整文本
        full_text = question_part + " Answer: " + answer_part
        
        # Tokenize完整文本
        tokens = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        # 找到"Answer:"在token序列中的位置
        answer_start_text = question_part + " Answer:"
        answer_start_tokens = self.tokenizer(
            answer_start_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        
        # 创建labels：只对answer部分计算loss
        labels = input_ids.clone()
        answer_start_pos = len(answer_start_tokens)
        
        # 将问题部分和"Answer:"部分设为IGNORE_INDEX
        labels[:answer_start_pos] = IGNORE_INDEX
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "index": index,
            "text_type": "qa",
            "answer_start_pos": answer_start_pos
        }

    def _process_paraphrased(self, text, index):
        """处理paraphrased格式 - 只对(?) :之后的部分计算loss"""
        if "(?) :" not in text:
            return self._process_statement(text, index)
        
        # 分割前缀和答案
        parts = text.split("(?) :", 1)
        prefix_part = parts[0].strip() + " (?) :"
        answer_part = parts[1].strip()
        
        # 构建完整文本
        full_text = prefix_part + answer_part
        
        # Tokenize完整文本
        tokens = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        # 找到"(?) :"在token序列中的位置
        prefix_tokens = self.tokenizer(
            prefix_part,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        
        # 创建labels：只对答案部分计算loss
        labels = input_ids.clone()
        answer_start_pos = len(prefix_tokens)
        
        # 将前缀部分设为IGNORE_INDEX
        labels[:answer_start_pos] = IGNORE_INDEX
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "index": index,
            "text_type": "paraphrased",
            "answer_start_pos": answer_start_pos
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample[self.text_key]
        index = sample["index"]
        
        # 根据文本类型选择处理方式
        if self._is_qa(text):
            return self._process_qa(text, index)
        elif self._is_paraphrased(text):
            return self._process_paraphrased(text, index)
        else:  # statement或其他格式
            return self._process_statement(text, index)
