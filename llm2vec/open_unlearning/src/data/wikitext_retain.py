import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from data.utils import add_dataset_index, preprocess_pretraining_instance

class WikiTextRetainDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        text_key="text",
        min_len=50,
        max_len=2000,
        max_length=512,
        predict_with_generate=False
    ):
        super(WikiTextRetainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.template_args = template_args
        self.max_length = max_length
        self.text_key = text_key
        self.min_len = min_len
        self.max_len = max_len
        self.predict_with_generate = predict_with_generate
        
        # Load WikiText dataset
        self.data = self._load_and_filter_wikitext()
        self.data = add_dataset_index(self.data)

    def _load_and_filter_wikitext(self):
        """加载并过滤WikiText数据"""
        # 加载wikitext-2-raw-v1的test split
        raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # 过滤数据
        filtered_data = []
        for item in raw_data:
            text = str(item[self.text_key]).strip()
            if self.min_len <= len(text) <= self.max_len:
                filtered_data.append({self.text_key: text})
        
        print(f"WikiText retain dataset: {len(filtered_data)} samples after filtering")
        
        # 转换为datasets格式
        from datasets import Dataset as HFDataset
        return HFDataset.from_list(filtered_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample[self.text_key]
        index = sample["index"]
        
        # 使用预训练格式处理
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            "",  # 没有prefix
            text,
            self.max_length,
            self.predict_with_generate
        )
        
        return {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"], 
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
            "original_text": text
        }
