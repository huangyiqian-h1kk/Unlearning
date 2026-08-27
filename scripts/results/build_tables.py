#!/usr/bin/env python3
"""Build deterministic archived-result tables without opening historical archives."""
import argparse,csv,json,math,pathlib

PROBE_ORDER=("ATT","IDeq","ID")
BLOCKING={"missing","unresolved"}
def clip100(x): return max(0.0,min(100.0,x))
def regime_a_generation_retain(s,b): return clip100(100*s/b)
def regime_a_generation_forget(s,b): return clip100(100*(1-s/b))
def regime_a_mcq_retain(s,b,p0=.25): return clip100(100*(s-p0)/(b-p0))
def regime_a_mcq_forget(s,b,p0=.25): return clip100(100*(1-(s-p0)/(b-p0)))
def regime_b_generation(s_retain,s_forget,b):
    r=100*s_retain/b; f=100*s_forget/b; return r,f,f-r
def regime_b_mcq(s_retain,s_forget,b,p0=.25):
    r=100*(s_retain-p0)/(b-p0); f=100*(s_forget-p0)/(b-p0); return r,f,f-r
def binomial_greater_pvalue(k,n,p0=.25):
    return sum(math.comb(n,i)*p0**i*(1-p0)**(n-i) for i in range(k,n+1))
def baseline_significant(k,n,p0=.25,alpha=.05): return binomial_greater_pvalue(k,n,p0)<alpha
def mask_if_not_significant(value,k,n,p0=.25,alpha=.05): return value if baseline_significant(k,n,p0,alpha) else "MASKED"
def row_average(values): return sum(v for k,v in values.items() if k!="MMLU" and isinstance(v,(int,float)))/sum(1 for k,v in values.items() if k!="MMLU" and isinstance(v,(int,float)))
def assert_probe_order(keys):
    observed=tuple(k for k in keys if k in PROBE_ORDER)
    if observed and observed != PROBE_ORDER[:len(observed)]: raise ValueError(f"probe order must be {PROBE_ORDER}, got {observed}")
def renderable(exp): return not (set(exp["status_flags"]) & BLOCKING)
def fmt(x): return "N/A" if x is None else "–" if x=="MASKED" else f"{x:.2f}"

def metric(snapshot,role,key): return snapshot.get("metrics",{}).get(role,{}).get(key)
def raw_columns(target):
    other="celebrity_deaths" if target=="diagnosis" else "celebrity_diagnosis"
    forget="celebrity_diagnosis" if target=="diagnosis" else "celebrity_deaths"
    mcq_r=("mcqs_deaths_att","mcqs_deaths_id_eq","mcqs_deaths_id_sim") if target=="diagnosis" else ("mcqs_diagnosis_att",None,"mcqs_diagnosis_id")
    mcq_f=("mcqs_diagnosis_att",None,"mcqs_diagnosis_id") if target=="diagnosis" else ("mcqs_deaths_att","mcqs_deaths_id_eq","mcqs_deaths_id_sim")
    return other,forget,mcq_r,mcq_f

def build_a(exp,snap,baseline,base_snap):
    cols={}; other,forget,mr,mf=raw_columns(exp["knowledge_target"]); role="metrics"
    for label,suffix,mode in (("R-QA","Greedy-QA","r"),("R-Cloze","Greedy-Cloze","r"),("R-BG","Greedy-BG-Probe","r"),("F-QA","Greedy-QA","f"),("F-Cloze","Greedy-Cloze","f"),("F-BG","Greedy-BG-Probe","f")):
        dom=other if mode=="r" else forget; s=metric(snap,role,f"{dom}-{suffix}"); b=metric(base_snap,role,f"{dom}-{suffix}")
        cols[label]=None if s is None or b in (None,0) else (regime_a_generation_retain(s,b) if mode=="r" else regime_a_generation_forget(s,b))
    for prefix,keys,mode in (("R",mr,"r"),("F",mf,"f")):
        for probe,key in zip(PROBE_ORDER,keys):
            s=metric(snap,role,f"MCQ-llh-{key}") if key else None; b=metric(base_snap,role,f"MCQ-llh-{key}") if key else None
            cols[f"{prefix}-{probe}"]=None if s is None or b in (None,.25) else (regime_a_mcq_retain(s,b) if mode=="r" else regime_a_mcq_forget(s,b))
    cols["MMLU"]=snap.get("mmlu",{}).get("accuracy") if snap.get("mmlu") else None
    vals={k:v for k,v in cols.items() if k!="MMLU"}; cols["Average"]=row_average(vals) if any(isinstance(v,(int,float)) for v in vals.values()) else None
    return cols

def pooled_baseline(base_snap,key):
    r=metric(base_snap,"retain_metrics",key); f=metric(base_snap,"forget_metrics",key)
    if r is None or f is None:return None
    # Archived generation splits contain 900 retain and 100 forget items; MCQ counts are unresolved.
    if key.startswith("PMC-Greedy-"): return (900*r+100*f)/1000
    return None
def build_b(exp,snap,base_snap):
    cols={}
    for label,key in (("QA","PMC-Greedy-QA"),("Cloze","PMC-Greedy-Cloze"),("BG","PMC-Greedy-BG-Probe")):
        b=pooled_baseline(base_snap,key); r=metric(snap,"retain_metrics",key); f=metric(snap,"forget_metrics",key)
        vals=(None,None,None) if None in (b,r,f) else regime_b_generation(r,f,b)
        for sub,val in zip(("R","F","Delta"),vals): cols[f"{label}-{sub}"]=val
    for probe,kr,kf in zip(PROBE_ORDER,("mcqs_PMC_retain_att","mcqs_PMC_retain_id_equal","mcqs_PMC_retain_id_identical"),("mcqs_PMC_forget_att","mcqs_PMC_forget_id_equal","mcqs_PMC_forget_id_identical")):
        # Exact pooled MCQ success/sample counts were not archived; significance and relative values stay unresolved.
        cols[f"{probe}-R"]=cols[f"{probe}-F"]=cols[f"{probe}-Delta"]=None
    cols["MMLU"]=snap.get("mmlu",{}).get("accuracy") if snap.get("mmlu") else None
    vals={k:v for k,v in cols.items() if k!="MMLU"}; cols["Average"]=row_average(vals) if any(isinstance(v,(int,float)) for v in vals.values()) else None
    return cols

def emit(path,rows):
    path.mkdir(parents=True,exist_ok=True); headers=["Experiment","Model","Method","Status"]+list(next(iter(rows))["values"] if rows else [])
    matrix=[[r["id"],r["model"],r["method"],";".join(r["status"])]+[fmt(r["values"].get(h)) for h in headers[4:]] for r in rows]
    with open(path/"table.csv","w",newline="",encoding="utf-8") as f: w=csv.writer(f,lineterminator="\n");w.writerow(headers);w.writerows(matrix)
    md=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+["| "+" | ".join(x)+" |" for x in matrix]
    (path/"table.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    latex=["\\begin{tabular}{"+"l"*len(headers)+"}"," & ".join(headers)+r" \\",r"\hline"]+[" & ".join(x).replace("_",r"\_")+r" \\" for x in matrix]+["\\end{tabular}"]
    (path/"table.tex").write_text("\n".join(latex)+"\n",encoding="utf-8")

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args(argv)
    man=json.load(open(a.manifest)); root=pathlib.Path(a.manifest).parent/"raw_metrics"; exps={e["id"]:e for e in man["experiments"]}; snaps={i:json.load(open(root/f"{i}.json")) for i in exps}; out=pathlib.Path(a.output_dir); reconc=[]
    for regime,target,dirname in (("A","diagnosis","regime_a_diagnosis"),("A","deaths","regime_a_deaths"),("B","pmc","regime_b_pmc")):
        rows=[]
        for e in man["experiments"]:
            if e["regime"]!=regime or e["knowledge_target"]!=target:continue
            values={}
            if renderable(e) and e.get("baseline_experiment_id"):
                b=exps[e["baseline_experiment_id"]]; values=build_a(e,snaps[e["id"]],b,snaps[b["id"]]) if regime=="A" else build_b(e,snaps[e["id"]],snaps[b["id"]])
            rows.append({"id":e["id"],"model":e["resolved_model_id"],"method":e["method"]+("‡" if "non_comparable" in e["status_flags"] else ""),"status":e["status_flags"],"values":values})
            reconc.append([e["id"],regime,target,";".join(e["status_flags"]),"; ".join(e.get("known_limitations",[]))])
        # Ensure consistent columns even when unresolved rows precede populated rows.
        template=next((r["values"] for r in rows if r["values"]),{})
        for r in rows:
            if not r["values"]: r["values"]={k:None for k in template}
        emit(out/dirname,rows)
    with open(out/"reconciliation.csv","w",newline="",encoding="utf-8") as f: w=csv.writer(f,lineterminator="\n");w.writerow(["experiment_id","regime","target","status_flags","limitations"]);w.writerows(reconc)
if __name__=="__main__":main()
