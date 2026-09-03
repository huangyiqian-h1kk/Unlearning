import json, csv, logging, tqdm, torch, random
from utils import answer_present, ensure_dir
from metrics import greedy_hit, mcq_hit
from prompt_builder import PromptBuilder
from probabilistic_extraction import score_answer_logprob  # 同前文
import os

class Evaluator:
    def __init__(self,cfg,tokenizer,model,qa_deaths,qa_diag,mcq_rows):
        self.cfg=cfg; self.tok=tokenizer; self.model=model
        self.prompt=PromptBuilder()
        self.qa_deaths=qa_deaths; self.qa_diag=qa_diag; self.mcq=mcq_rows
        self.records={}; self.probs={}; self.topk={}

        ensure_dir(cfg["logging"]["out_dir"])
        self.sample_file=cfg["logging"]["out_dir"]+"/"+cfg["logging"]["sample_jsonl"]
        if cfg["logging"]["overwrite"] and os.path.exists(self.sample_file):
            os.remove(self.sample_file)

    # ----- Generation helpers -----
    def _gen(self,p,max_new=None,do_sample=False,temp=None,top_p=None):
        inputs=self.tok(p,return_tensors="pt").to(self.model.device)
        out=self.model.generate(**inputs,
            max_new_tokens=max_new or self.cfg["generation"]["max_new_tokens"],
            do_sample=do_sample,
            temperature=temp or self.cfg["generation"]["temperature"],
            top_p=top_p or self.cfg["generation"]["top_p"],
            pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0],skip_special_tokens=True)

    def _gen_k(self,p,k,temp,top_p):
        inputs=self.tok(p,return_tensors="pt").to(self.model.device)
        outs=self.model.generate(**inputs,
            max_new_tokens=self.cfg["generation"]["max_new_tokens"],
            do_sample=True, temperature=temp, top_p=top_p,
            num_return_sequences=k,
            pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o,skip_special_tokens=True) for o in outs]

    # ----- record utils -----
    def r(self,key,val): self.records.setdefault(key,[]).append(val)
    def rp(self,key,val): self.probs.setdefault(key,[]).append(val)
    def rt(self,key,val): self.topk.setdefault(key,[]).append(val)

    def log_sample(self,row,task,prompt,gen,hit):
        with open(self.sample_file,"a",encoding="utf-8") as f:
            f.write(json.dumps({
                **row, "eval_task":task,"prompt":prompt,
                "generation":gen,"hit":hit
            },ensure_ascii=False)+"\n")

    # ----- Main evaluate -----
    def run(self):
        # 1. Greedy tasks for QA sets
        for row in tqdm.tqdm(self.qa_deaths+self.qa_diag,desc="Greedy QA"):
            for tag,p in self.prompt.greedy_prompts(row).items():
                gen=self._gen(p)
                hit=greedy_hit(gen,row["answer value"]); self.r(tag,hit)
                if self.cfg["prob_extraction"]["enabled"]:
                    lp=score_answer_logprob(self.model,self.tok,p,row["answer value"],
                                            self.cfg["prob_extraction"]["method"])
                    self.rp(tag,lp)
                if self.cfg["topk_probe"]["enabled"]:
                    seqs=self._gen_k(p,self.cfg["topk_probe"]["k"],
                                     self.cfg["topk_probe"]["temperature"],
                                     self.cfg["topk_probe"]["top_p"])
                    leak=int(any(answer_present(s,row["answer value"]) for s in seqs))
                    self.rt(tag,leak)
                self.log_sample(row,tag,p,gen,hit)

        # 2. MCQ tasks
        for row in tqdm.tqdm(self.mcq,desc="MCQ"):
            p=row["prompt"]; correct_letter=row["correct_letter"]
            gen=self._gen(p,max_new=4)
            hit=mcq_hit(gen,correct_letter)
            tag=f"{row['domain']}__{row['task']}__{row.get('mcq_type','equal')}"
            self.r(tag,hit); self.log_sample(row,tag,p,gen,hit)

        return self._summarize()

    def _summarize(self):
        summary={}
        for k,v in self.records.items():
            summary[k]={"acc":sum(v)/len(v),"n":len(v)}
            if k in self.probs: summary[k]["prob"] = sum(self.probs[k])/len(self.probs[k])
            if k in self.topk:  summary[k]["topk_leak"] = sum(self.topk[k])/len(self.topk[k])

        # save
        path_json=self.cfg["logging"]["out_dir"]+"/"+self.cfg["logging"]["summary_json"]
        path_csv =self.cfg["logging"]["out_dir"]+"/"+self.cfg["logging"]["summary_csv"]
        json.dump(summary,open(path_json,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
        with open(path_csv,"w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f); writer.writerow(["task","acc","prob","topk_leak","n"])
            for k,v in summary.items():
                writer.writerow([k,v["acc"],v.get("prob",""),v.get("topk_leak",""),v["n"]])
        return summary

