from utils import load_jsonl
def load_all_mcq(mcq_paths):
    mcq=[]
    for p in mcq_paths:
        mcq.extend(load_jsonl(p))
    return mcq

