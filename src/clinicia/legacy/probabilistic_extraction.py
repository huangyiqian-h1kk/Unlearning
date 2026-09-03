import torch, math, numpy as np
from utils import normalize

def score_answer_logprob(model,tok,prompt,ans,method="avg_logprob"):
    ans=normalize(ans); 
    if not ans: return float("nan")
    with torch.no_grad():
        p_ids=tok(prompt,return_tensors="pt").to(model.device)
        a_ids=tok(ans,add_special_tokens=False)["input_ids"]
        ids=torch.tensor([p_ids["input_ids"][0].tolist()+a_ids],device=model.device)
        attn=torch.ones_like(ids)
        logits=model(ids,attention_mask=attn).logits  # (1,L,V)
        start=len(p_ids["input_ids"][0])
        lp=[]
        for i,tok_id in enumerate(a_ids):
            pos=start+i-1
            l=torch.log_softmax(logits[0,pos],dim=-1)[tok_id].item()
            lp.append(l)
        if not lp: return float("nan")
        if method=="avg_logprob": return sum(lp)/len(lp)
        if method=="prod": return sum(lp)          # logprod
        if method=="min": return min(lp)
        return sum(lp)/len(lp)

