import torch, sys, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------ 模型加载 ------------------------ #
def load_model(model_path: str):
    try:
        cfg = PeftConfig.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path).to(device)
        model = PeftModel.from_pretrained(base, model_path).merge_and_unload()
        print("✅ LoRA merged model loaded.")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        print("✅ Standard HF model loaded.")
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token
    return tok, model

# ------------------------ Prompt 构造 --------------------- #
def build_prompts(samples, task):
    res = []
    for s in samples:
        if task == "Greedy-QA":
            print(s['answer key'])
            p = f"As far as we remember, what is the {s['answer key']} of the patient with id {s['question value']}?"
        elif task == "Greedy-Cloze":
            p = f"As far as we remember, the {s['answer key']} of the person with ID {s['question value']} is"
        elif task == "Greedy-BG-Probe":
            bg = s.get("short intro") or s.get("archive_wiout_ans", "")
            p = f"Here is the archive of a patient with ID {s['question value']}, we have seen it many times:  {bg} Can you recall this person's {s['answer key']} "
        else:
            p = ""
        res.append(p)
    return res

# ------------------------ Batch 生成 ---------------------- #
def batch_generate(tok, mdl, prompts, bs, n_tok, chat_tmpl=False):
    if chat_tmpl and hasattr(tok, "apply_chat_template"):
        inputs = [tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True) for p in prompts]
    else:
        inputs = prompts

    dl = DataLoader(inputs, batch_size=bs)
    out = []
    mdl.eval()
    for batch in tqdm(dl, desc="Generating"):
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        with torch.no_grad():
            gens = mdl.generate(**enc, max_new_tokens=n_tok)
        out.extend(tok.batch_decode(gens, skip_special_tokens=True))
    return out

# ------------------- MCQ 解析式提取 ---------------------- #
letter_pat = re.compile(r"\b([A-J])[\)|）:：．。]?", re.I)
def extract_letter(output: str, mapping: Dict[str, str]):
    m = letter_pat.search(output)
    if m:
        ltr = m.group(1).upper()
        if ltr in mapping:
            return ltr
    # fallback by content
    for ltr, txt in mapping.items():
        if txt.lower() in output.lower():
            return ltr
    return None

# ------------------- MCQ Loglikelihood ------------------- #
def mcq_llh(tok, mdl, prompt: str, mapping: Dict[str, str],
            bs=8, chat_tmpl=False, n_tok_ctx_cache=None):
    """
    返回得分最高的 letter 及所有得分
    """
    # 先缓存 prompt 编码长度，减少重复 encode
    if n_tok_ctx_cache is None or prompt not in n_tok_ctx_cache:
        ctx_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        n_tok_ctx_cache[prompt] = ctx_ids.size(1)
    ctx_len = n_tok_ctx_cache[prompt]

    scores = {}
    for ltr, txt in mapping.items():
        full = f"{prompt}{ltr}) {txt}"
        full_ids = tok(full, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = mdl(full_ids)
        logits = out.logits  # [1, seq, vocab]

        # 计算 choice 部分 logprob
        llh = 0.0
        for i in range(ctx_len - 1, full_ids.size(1) - 1):
            next_id = full_ids[0, i + 1]
            llh += torch.log_softmax(logits[0, i], dim=-1)[next_id].item()
        scores[ltr] = llh
    best = max(scores, key=scores.get)
    return best, scores

# ------------------------- 主逻辑 ------------------------ #
def main(cfg_path: str):
    cfg = json.load(open(cfg_path))
    tok, mdl = load_model(cfg["model_path"])
    bs, n_tok = cfg.get("batch_size", 8), cfg.get("max_new_tokens", 64)

    chat_tmpl = cfg.get("apply_chat_template")
    if chat_tmpl is None:
        chat_tmpl = hasattr(tok, "apply_chat_template") and "instruct" in tok.name_or_path.lower()

    mcq_mode = cfg.get("mcq_eval_mode", "generate")  # generate / llh / both

    results, logs = {}, []
    out_dir = Path(cfg.get("output_dir", cfg["model_path"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    greedy_tasks = ["Greedy-QA", "Greedy-Cloze", "Greedy-BG-Probe"]

    # -------- Greedy 类 -------- #
    for domain, path in cfg["evaluation_sets"].items():
        ds = load_dataset("json", data_files=path)["train"]
        ds = [d for d in ds if "answer key" in d and "question value" in d]

        for task in greedy_tasks:
            print(f"\n🔍 {domain} - {task}")
            prompts = build_prompts(ds, task)
            outs = batch_generate(tok, mdl, prompts, bs, n_tok, chat_tmpl)
            hit = 0
            for s, pmt, out in zip(ds, prompts, outs):
                ans = s["answer value"] 
                if "'']]" in ans:
                    ans = str(int(eval(ans)[0][0]))
                ok = s["answer value"].lower() in out.lower()
                hit += ok
                logs.append(dict(task=task, domain=domain, prompt=pmt,
                                 gold=ans, output=out.strip(), correct=ok))
            acc = hit / len(ds)
            results[f"{domain}-{task}"] = acc
            print(f"✅ Acc {acc:.4f}")

    # -------- MCQ 类 -------- #
    if mcq_mode.lower() in {"generate", "both"}:
        for name, path in cfg["mcq_sets"].items():
            samples = load_dataset("json", data_files=path)["train"]
            prompts = [s["prompt"] for s in samples]
            outs = batch_generate(tok, mdl, prompts, bs, n_tok, chat_tmpl)
            hit = 0
            for s, out in zip(samples, outs):
                pred = extract_letter(out, s["mapping"])
                ok = pred == s["correct_letter"]
                hit += ok
                logs.append(dict(task=f"MCQ-gen-{name}", prompt=s["prompt"],
                                 output=out.strip(), pred=pred,
                                 correct_letter=s["correct_letter"], correct=ok))
            acc = hit / len(samples)
            results[f"MCQ-gen-{name}"] = acc
            print(f"✅ MCQ-gen-{name} Acc {acc:.4f}")

    if mcq_mode.lower() in {"llh", "both"}:
        cache_ctx_len = {}
        for name, path in cfg["mcq_sets"].items():
            samples = load_dataset("json", data_files=path)["train"]
            hit = 0
            for s in tqdm(samples, desc=f"LLH-MCQ {name}"):
                pred, sc = mcq_llh(tok, mdl, s["prompt"],
                                   s["mapping"], chat_tmpl=chat_tmpl,
                                   n_tok_ctx_cache=cache_ctx_len)
                ok = pred == s["correct_letter"]
                hit += ok
                logs.append(dict(task=f"MCQ-llh-{name}", prompt=s["prompt"],
                                 loglikelihood_scores=sc, pred=pred,
                                 correct_letter=s["correct_letter"], correct=ok))
            acc = hit / len(samples)
            results[f"MCQ-llh-{name}"] = acc
            print(f"✅ MCQ-llh-{name} Acc {acc:.4f}")

    # -------- 保存 -------- #
    json.dump(results, open(out_dir / "evaluation_results.json", "w"), indent=2)
    with open(out_dir / "detailed_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in logs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n📑 Summary & logs saved to {out_dir}")
    print("🏁 Done.")

# ------------------------ CLI ----------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])

