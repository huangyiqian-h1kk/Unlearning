from torch.utils.data import Dataset

from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
    preprocess_instruction_instance,
)

IGNORE_INDEX = -100

class PMCFinetuneDataset(Dataset):
    """Dataset for finetuning on PMC_QA style data (multi-attribute per sample)."""

    def __init__(
        self, hf_args, template_args, tokenizer, max_length=2048, include=None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include = include or []
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.flattened_data = self._preprocess_all()

    def _preprocess_all(self):
        flattened = []
        for sample in self.data:
            index = sample["index"]

            # QA format from dialogue
            if "dialogue" in self.include and sample.get("Medical Dialogue Question"):
                question = sample["Medical Dialogue Question"]
                answer = sample.get("Medical Dialogue Answer", "")
                qa_item = preprocess_instruction_instance(
                    self.tokenizer,
                    instruction=question,
                    output=answer,
                    max_length=self.max_length,
                )
                qa_item["index"] = index
                flattened.append(qa_item)

            # LM-style fields
            for field in ["note", "patient", "title", "archive"]:
                if field in self.include and sample.get(field):
                    if field == "archive":
                        value = sample["archive"]
                        text = "Patient Archive:\n"
                        text += "\n".join(f"{k}: {v}" for k, v in value.items())
                    else:
                        text = sample[field]

                    lm_item = preprocess_pretraining_instance(
                        self.tokenizer,
                        prompt="",
                        answer=text,
                        max_length=self.max_length,
                    )
                    lm_item["index"] = index
                    flattened.append(lm_item)

        return flattened

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        return self.flattened_data[idx]

