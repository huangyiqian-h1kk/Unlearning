import json, os, random, re, unicodedata, torch, numpy as np
from typing import List, Dict

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path:str): os.makedirs(path, exist_ok=True)

def load_jsonl(path:str)->List[Dict]:
    with open(path,encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def normalize(txt:str)->str:
    return unicodedata.normalize("NFKC",(txt or "")).strip()

def answer_present(gen:str, ans:str)->bool:
    g=normalize(gen).lower(); a=normalize(ans).lower()
    if a in g: return True
    # token overlap
    atoks=[t for t in re.split(r"\W+",a) if t]
    gtoks=set(re.split(r"\W+",g))
    return sum(t in gtoks for t in atoks)/len(atoks) >= 0.6 if atoks else False

