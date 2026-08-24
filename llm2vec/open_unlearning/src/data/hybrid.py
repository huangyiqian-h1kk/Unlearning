import torch
from datasets import load_dataset
from torch.utils.data import Dataset

IGNORE_INDEX = -100

class HybridQADataset(Dataset):
    def __init__(
        self,
        hf_args,
        tokenizer,
        max_length=512,
        template_args=None,
        split=None,
        **kwargs
    ):
        self.data = load_dataset(
            hf_args["path"],
            data_files=hf_args["data_files"],
            split=hf_args.get("split", split) or "train"
        )
        print(f"[DEBUG] Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"[DEBUG] Pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        print(f"[DEBUG] EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")


        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw = list(self.data[idx].values())[0].strip()

        if "Question:" in raw and "Answer:" in raw:
            q, a = raw.split("Answer:", 1)
            prompt = q.strip()
            answer = a.strip()

        elif "?" in raw:
            q, a = raw.split("?", 1)
            prompt = q.strip() + "?"
            answer = a.strip()

        else:
            prompt = ""
            answer = raw.strip()
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        input_ids = prompt_ids + answer_ids
        input_ids = input_ids[:self.max_length]

        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            }

