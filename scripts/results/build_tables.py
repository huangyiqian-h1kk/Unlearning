#!/usr/bin/env python3
"""Build deterministic tables solely from manifest metadata and compact snapshots."""
import argparse
import csv
import json
import math
import pathlib

PROBE_ORDER = ("ATT", "IDeq", "ID")
BLOCKING = {"missing", "unresolved"}
MASKED = "MASKED"

def clip100(value): return max(0.0, min(100.0, value))
def regime_a_generation_retain(score, baseline): return clip100(100 * score / baseline)
def regime_a_generation_forget(score, baseline): return clip100(100 * (1 - score / baseline))
def regime_a_mcq_retain(score, baseline, p0=.25): return clip100(100 * (score-p0)/(baseline-p0))
def regime_a_mcq_forget(score, baseline, p0=.25): return clip100(100 * (1-(score-p0)/(baseline-p0)))
def regime_b_generation(retain, forget, baseline):
    r=100*retain/baseline;f=100*forget/baseline;return r,f,f-r
def regime_b_mcq(retain, forget, baseline, p0=.25):
    r=100*(retain-p0)/(baseline-p0);f=100*(forget-p0)/(baseline-p0);return r,f,f-r
def binomial_greater_pvalue(k,n,p0=.25): return sum(math.comb(n,i)*p0**i*(1-p0)**(n-i) for i in range(k,n+1))
def baseline_significant(k,n,p0=.25,alpha=.05): return binomial_greater_pvalue(k,n,p0)<alpha
def row_average(values, regime=None):
    selected=[value for key,value in values.items() if key!="MMLU" and not (regime=="B" and key.endswith("-Delta")) and isinstance(value,(int,float))]
    return sum(selected)/len(selected) if selected else None
def assert_probe_order(keys):
    observed=tuple(key for key in keys if key in PROBE_ORDER)
    if observed and observed!=PROBE_ORDER[:len(observed)]:raise ValueError(f"probe order must be {PROBE_ORDER}, got {observed}")
def renderable(experiment): return not (set(experiment["status_flags"]) & BLOCKING)
def metric(snapshot,role,key): return snapshot.get("metrics",{}).get(role,{}).get(key)
def cell_allowed(experiment,cell): return not (set(experiment.get("cell_status",{}).get(cell,[])) & BLOCKING)

def raw_columns(target):
    if PROBE_ORDER != ("ATT", "IDeq", "ID"):
        raise ValueError("canonical probe order was modified")
    other="celebrity_deaths" if target=="diagnosis" else "celebrity_diagnosis";forget="celebrity_diagnosis" if target=="diagnosis" else "celebrity_deaths"
    retain=("mcqs_deaths_att","mcqs_deaths_id_eq","mcqs_deaths_id_sim") if target=="diagnosis" else ("mcqs_diagnosis_att",None,"mcqs_diagnosis_id")
    forgotten=("mcqs_diagnosis_att",None,"mcqs_diagnosis_id") if target=="diagnosis" else ("mcqs_deaths_att","mcqs_deaths_id_eq","mcqs_deaths_id_sim")
    assert_probe_order(PROBE_ORDER);return other,forget,retain,forgotten

def build_a(experiment,snapshot,baseline_snapshot):
    values={};daggers=set();other,forget,retain_keys,forget_keys=raw_columns(experiment["knowledge_target"])
    for label,suffix,mode in (("R-QA","Greedy-QA","r"),("R-Cloze","Greedy-Cloze","r"),("R-BG","Greedy-BG-Probe","r"),("F-QA","Greedy-QA","f"),("F-Cloze","Greedy-Cloze","f"),("F-BG","Greedy-BG-Probe","f")):
        domain=other if mode=="r" else forget;s=metric(snapshot,"metrics",f"{domain}-{suffix}");b=metric(baseline_snapshot,"metrics",f"{domain}-{suffix}")
        values[label]=None if s is None or b in (None,0) else regime_a_generation_retain(s,b) if mode=="r" else regime_a_generation_forget(s,b)
    for prefix,keys,mode,domain in (("R",retain_keys,"r",other),("F",forget_keys,"f",forget)):
        for probe,key in zip(PROBE_ORDER,keys):
            cell=f"{prefix}-{probe}"
            if key is None:values[cell]=None;continue
            significance=baseline_snapshot["significance"][f"{domain}:{probe}"]
            if not significance["significant"]:
                values[cell]=MASKED
                if experiment["method"]=="Baseline":daggers.add(cell)
                continue
            s=metric(snapshot,"metrics","MCQ-llh-"+key);b=significance["accuracy"]
            values[cell]=regime_a_mcq_retain(s,b) if mode=="r" else regime_a_mcq_forget(s,b)
    values["MMLU"]=snapshot.get("mmlu",{}).get("accuracy") if snapshot.get("mmlu") else None
    for cell in tuple(values):
        if not cell_allowed(experiment,cell):values[cell]=None
    values["Average"]=row_average(values,"A");return values,daggers

def pooled_generation(baseline_snapshot,key):
    return (900*metric(baseline_snapshot,"retain_metrics",key)+100*metric(baseline_snapshot,"forget_metrics",key))/1000

def build_b(experiment,snapshot,baseline_snapshot):
    values={};daggers=set()
    for label,key in (("QA","PMC-Greedy-QA"),("Cloze","PMC-Greedy-Cloze"),("BG","PMC-Greedy-BG-Probe")):
        triplet=regime_b_generation(metric(snapshot,"retain_metrics",key),metric(snapshot,"forget_metrics",key),pooled_generation(baseline_snapshot,key))
        for suffix,value in zip(("R","F","Delta"),triplet):values[f"{label}-{suffix}"]=value
    rkeys=("mcqs_PMC_retain_att","mcqs_PMC_retain_id_equal","mcqs_PMC_retain_id_identical");fkeys=("mcqs_PMC_forget_att","mcqs_PMC_forget_id_equal","mcqs_PMC_forget_id_identical")
    assert_probe_order(PROBE_ORDER)
    for probe,rkey,fkey in zip(PROBE_ORDER,rkeys,fkeys):
        significance=baseline_snapshot["significance"][f"pmc:{probe}"]
        cells=(f"{probe}-R",f"{probe}-F",f"{probe}-Delta")
        if not significance["significant"]:
            for cell in cells:
                values[cell]=MASKED
                if experiment["method"]=="Baseline":daggers.add(cell)
            continue
        triplet=regime_b_mcq(metric(snapshot,"retain_metrics","MCQ-llh-"+rkey),metric(snapshot,"forget_metrics","MCQ-llh-"+fkey),significance["accuracy"])
        for cell,value in zip(cells,triplet):values[cell]=value
    values["MMLU"]=snapshot.get("mmlu",{}).get("accuracy") if snapshot.get("mmlu") else None
    for cell in tuple(values):
        if not cell_allowed(experiment,cell):values[cell]=None
    values["Average"]=row_average(values,"B");return values,daggers

def display(value,dagger=False,latex=False):
    if value is None:return r"\mathrm{N/A}" if latex else "N/A"
    if value==MASKED:
        result=r"\textemdash{}" if latex else "–"
        return result+(r"\textsuperscript{\dagger}" if latex and dagger else "†" if dagger else "")
    result=f"{value:.2f}"
    if dagger:return result+(r"\textsuperscript{\dagger}" if latex else "†")
    return result

def emit(directory,rows):
    directory.mkdir(parents=True,exist_ok=True);headers=["Experiment","Model","Method","Status"]+list(rows[0]["values"])
    matrix=[]
    for row in rows:
        matrix.append([row["id"],row["model"],row["method"],";".join(row["status"])]+[display(row["values"][key],key in row["daggers"]) for key in headers[4:]])
    with open(directory/"table.csv","w",newline="",encoding="utf-8") as stream:writer=csv.writer(stream,lineterminator="\n");writer.writerow(headers);writer.writerows(matrix)
    markdown=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+["| "+" | ".join(row)+" |" for row in matrix]
    (directory/"table.md").write_text("\n".join(markdown)+"\n",encoding="utf-8")
    latex=[r"\begin{tabular}{"+"l"*len(headers)+"}"," & ".join(headers)+r" \\",r"\hline"]
    for row,rendered in zip(rows,matrix):
        prefix=[item.replace("_",r"\_").replace("‡",r"\textsuperscript{\ddagger}") for item in rendered[:4]];suffix=[display(row["values"][key],key in row["daggers"],True) for key in headers[4:]];latex.append(" & ".join(prefix+suffix)+r" \\")
    latex.append(r"\end{tabular}");(directory/"table.tex").write_text("\n".join(latex)+"\n",encoding="utf-8")

def raw_source_refs(snapshot):
    return ";".join(source.get("member",source.get("path","")) for source in snapshot["sources"] if source["role"] in {"metrics","retain_metrics","forget_metrics"})

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",required=True);parser.add_argument("--output-dir",required=True);args=parser.parse_args(argv)
    manifest=json.load(open(args.manifest));root=pathlib.Path(args.manifest).parent/"raw_metrics";experiments={item["id"]:item for item in manifest["experiments"]};snapshots={key:json.load(open(root/f"{key}.json")) for key in experiments};output=pathlib.Path(args.output_dir);reconciliation=[]
    for regime,target,name in (("A","diagnosis","regime_a_diagnosis"),("A","deaths","regime_a_deaths"),("B","pmc","regime_b_pmc")):
        rows=[]
        for experiment in manifest["experiments"]:
            if experiment["regime"]!=regime or experiment["knowledge_target"]!=target:continue
            baseline=experiments[experiment["baseline_experiment_id"]]
            if renderable(experiment):values,daggers=(build_a(experiment,snapshots[experiment["id"]],snapshots[baseline["id"]]) if regime=="A" else build_b(experiment,snapshots[experiment["id"]],snapshots[baseline["id"]]))
            else:
                template=build_a(baseline,snapshots[baseline["id"]],snapshots[baseline["id"]])[0] if regime=="A" else build_b(baseline,snapshots[baseline["id"]],snapshots[baseline["id"]])[0];values={key:None for key in template};daggers=set()
            method=experiment["method"]+("‡" if "non_comparable" in experiment["status_flags"] else "");row={"id":experiment["id"],"model":experiment.get("intended_model_id") or experiment.get("resolved_model_id") or experiment["intended_paper_label"],"method":method,"status":experiment["status_flags"],"values":values,"daggers":daggers};rows.append(row)
            for cell,value in values.items():
                flags=experiment.get("cell_status",{}).get(cell,experiment["status_flags"]);reconciliation.append([experiment["id"],regime,target,row["model"],experiment["method"],cell,"",display(value,cell in daggers),"",";".join(flags),raw_source_refs(snapshots[experiment["id"]]),"","preserve archived evidence and status"])
            if "non_comparable" in experiment["status_flags"]:reconciliation.append([experiment["id"],regime,target,row["model"],experiment["method"],"starting_checkpoint","","","","non_comparable",raw_source_refs(snapshots[experiment["id"]]),"different starting checkpoint","display ‡; do not silently normalize"])
        emit(output/name,rows)
    headers=("experiment_id","regime","target","model","method","cell","archived_raw_value","reconstructed_value","reported_manuscript_value","status_flags","source_reference","discrepancy_description","release_treatment")
    with open(output/"reconciliation.csv","w",newline="",encoding="utf-8") as stream:writer=csv.writer(stream,lineterminator="\n");writer.writerow(headers);writer.writerows(reconciliation)

if __name__=="__main__":main()
