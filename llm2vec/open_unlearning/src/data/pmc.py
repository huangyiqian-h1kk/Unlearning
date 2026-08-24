from torch.utils.data import Dataset

from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
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
            pid = sample.get("question value", "unknown patient ID")

            # QA format from dialogue
            if "dialogue" in self.include and sample.get("Medical Dialogue Question"):
                question = sample["Medical Dialogue Question"]
                answer = sample.get("Medical Dialogue Answer", "")
                text = f"Question: {question}\nAnswer: {answer}"
                qa_item = preprocess_pretraining_instance(
                    self.tokenizer,
                    prefix="",
                    text_content=text,
                    max_length=self.max_length,
                )
                qa_item["index"] = index
                flattened.append(qa_item)

            # LM-style fields with identifier prefix
            for field in ["note", "patient", "title", "archive"]:
                if field in self.include and sample.get(field):
                    if field == "archive":
                        value = sample["archive"]
                        archive_str = "\n".join(f"{k}: {v}" for k, v in value.items())
                        content = f"Archive summary for patient ID {pid}:\n{archive_str}"
                    elif field == "note":
                        content = f"Note for patient ID {pid}:\n{sample[field]}"
                    elif field == "patient":
                        content = f"Patient description for patient ID {pid}: {sample[field]}"
                    elif field == "title":
                        content = f"Document title for patient ID {pid}: {sample[field]}"
                    else:
                        content = sample[field]

                    lm_item = preprocess_pretraining_instance(
                        self.tokenizer,
                        prefix="",
                        text_content=content,
                        max_length=self.max_length,
                    )
                    lm_item["index"] = index
                    flattened.append(lm_item)

        return flattened

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        return self.flattened_data[idx]

