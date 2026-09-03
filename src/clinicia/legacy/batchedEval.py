import torch
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    try:
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to(device)
        model = PeftModel.from_pretrained(base_model, model_path).to(device)
        model = model.merge_and_unload()
        print("✅ LoRA merged model loaded.")
    except:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        print("✅ Standard HuggingFace model loaded.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def batch_generate(tokenizer, model, prompts, batch_size, max_new_tokens):
    dataloader = DataLoader(prompts, batch_size=batch_size)
    results = []
    model.eval()
    for batch in tqdm(dataloader, desc="Generating"):
        inputs = tokenizer(batch, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

def construct_prompts(samples, task_type):
    prompts = []
    for s in samples:
        if task_type == "Greedy-QA":
            prompt = f"What is the {s['answer key']} of {s['question value']}?"
        elif task_type == "Greedy-Cloze":
            prompt = f"{s['question value']}'s {s['answer key']} is"
        elif task_type == "Greedy-BG-Probe":
            bg = s.get("short intro") or s.get("archive_wiout_ans", "")
            prompt = f"Let's talk about {s['question value']}. {bg} Tell me about the {s['answer key']} you know about this."
        else:
            prompt = ""
        prompts.append(prompt)
    return prompts

def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    tokenizer, model = load_model(cfg["model_path"])
    batch_size = cfg.get("batch_size", 8)
    max_new_tokens = cfg.get("max_new_tokens", 64)
    tasks = ["Greedy-QA", "Greedy-Cloze", "Greedy-BG-Probe"]
    results = {}
    detailed_records = []

    # === 确定输出目录 ===
    output_dir = cfg.get("output_dir", cfg["model_path"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # === Greedy 评估（含记录）
    for domain, path in cfg["evaluation_sets"].items():
        samples = load_dataset("json", data_files=path)["train"]
        samples = [s for s in samples if "answer key" in s and "question value" in s]
        for task in tasks:
            print(f"🔍 Evaluating {domain} - {task}")
            prompts = construct_prompts(samples, task)
            outputs = batch_generate(tokenizer, model, prompts, batch_size, max_new_tokens)

            correct_count = 0
            for s, prompt, out in zip(samples, prompts, outputs):
                correct = s["answer value"].lower() in out.lower()
                if correct:
                    correct_count += 1
                detailed_records.append({
                    "domain": domain,
                    "task": task,
                    "prompt": prompt,
                    "gold": s["answer value"],
                    "output": out,
                    "correct": correct
                })

            accuracy = correct_count / len(samples)
            results[f"{domain}-{task}"] = accuracy
            print(f"✅ {domain}-{task} Accuracy: {accuracy:.4f}")

    # === MCQ 评估（含记录）
    for name, path in cfg["mcq_sets"].items():
        print(f"🧠 Evaluating MCQ {name}")
        samples = load_dataset("json", data_files=path)["train"]
        prompts = [s["prompt"] for s in samples]
        outputs = batch_generate(tokenizer, model, prompts, batch_size, max_new_tokens)

        correct_count = 0
        for s, out in zip(samples, outputs):
            pred = out.strip()[0].upper()
            correct = pred == s["correct_letter"]
            if correct:
                correct_count += 1
            detailed_records.append({
                "task": f"MCQ-{name}",
                "prompt": s["prompt"],
                "output": out.strip(),
                "correct_letter": s["correct_letter"],
                "predicted_letter": pred,
                "correct": correct
            })

        accuracy = correct_count / len(samples)
        results[f"MCQ-{name}"] = accuracy
        print(f"✅ MCQ {name} Accuracy: {accuracy:.4f}")

    # === 保存评估结果 ===
    output_file = Path(output_dir) / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    detailed_file = Path(output_dir) / "detailed_outputs.jsonl"
    with open(detailed_file, "w", encoding="utf-8") as f:
        for rec in detailed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n📄 Saved summary to {output_file}")
    print(f"📄 Saved detailed logs to {detailed_file}")
    print(f"🎉 Evaluation complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval.py <config.json>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)

