#!/usr/bin/env python3
"""Build deterministic tables solely from the manifest and compact snapshots."""
import argparse
import csv
import json
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
    r=100*retain/baseline; f=100*forget/baseline; return r,f,f-r
def regime_b_mcq(retain, forget, baseline, p0=.25):
    r=100*(retain-p0)/(baseline-p0); f=100*(forget-p0)/(baseline-p0); return r,f,f-r
def row_average(values, regime=None):
    selected=[value for key,value in values.items() if key!="MMLU" and not (regime=="B" and key.endswith("-Delta")) and isinstance(value,(int,float))]
    return sum(selected)/len(selected) if selected else None
def assert_probe_order(keys):
    observed=tuple(key for key in keys if key in PROBE_ORDER)
    if observed and observed!=PROBE_ORDER[:len(observed)]: raise ValueError(f"probe order must be {PROBE_ORDER}, got {observed}")
def renderable(experiment): return not (set(experiment["status_flags"]) & BLOCKING)
def metric(snapshot,role,key): return snapshot.get("metrics",{}).get(role,{}).get(key)
def cell_allowed(experiment,cell): return not (set(experiment.get("cell_status",{}).get(cell,[])) & BLOCKING)


def raw_columns(target):
    if PROBE_ORDER != ("ATT", "IDeq", "ID"): raise ValueError("canonical probe order was modified")
    other="celebrity_deaths" if target=="diagnosis" else "celebrity_diagnosis"
    forget="celebrity_diagnosis" if target=="diagnosis" else "celebrity_deaths"
    retain=("mcqs_deaths_att","mcqs_deaths_id_eq","mcqs_deaths_id_sim") if target=="diagnosis" else ("mcqs_diagnosis_att",None,"mcqs_diagnosis_id")
    forgotten=("mcqs_diagnosis_att",None,"mcqs_diagnosis_id") if target=="diagnosis" else ("mcqs_deaths_att","mcqs_deaths_id_eq","mcqs_deaths_id_sim")
    assert_probe_order(PROBE_ORDER); return other,forget,retain,forgotten


def build_a(experiment,snapshot,baseline_snapshot):
    values={}; evidence={}; daggers=set(); other,forget,retain_keys,forget_keys=raw_columns(experiment["knowledge_target"])
    for label,suffix,mode in (("R-QA","Greedy-QA","r"),("R-Cloze","Greedy-Cloze","r"),("R-BG","Greedy-BG-Probe","r"),("F-QA","Greedy-QA","f"),("F-Cloze","Greedy-Cloze","f"),("F-BG","Greedy-BG-Probe","f")):
        domain=other if mode=="r" else forget; key=f"{domain}-{suffix}"; s=metric(snapshot,"metrics",key); b=metric(baseline_snapshot,"metrics",key)
        values[label]=None if s is None or b in (None,0) else regime_a_generation_retain(s,b) if mode=="r" else regime_a_generation_forget(s,b)
        if s is not None and b is not None: evidence[label]={"method_raw_score":s,"baseline_raw_score":b,"formula":"regime_a_generation_retain" if mode=="r" else "regime_a_generation_forget"}
    for prefix,keys,mode,domain in (("R",retain_keys,"r",other),("F",forget_keys,"f",forget)):
        for probe,key in zip(PROBE_ORDER,keys):
            cell=f"{prefix}-{probe}"
            if key is None: values[cell]=None; continue
            significance=baseline_snapshot["significance"][f"{domain}:{probe}"]; s=metric(snapshot,"metrics","MCQ-llh-"+key)
            evidence[cell]={"method_raw_accuracy":s,"baseline_raw_accuracy":significance["accuracy"],"dataset_count":significance["sample_count"],"baseline_success_count":significance["success_count"],"exact_pvalue":significance["pvalue"],"significant":significance["significant"],"formula":"regime_a_mcq_retain" if mode=="r" else "regime_a_mcq_forget"}
            if not significance["significant"]:
                values[cell]=MASKED
                if experiment["method"]=="Baseline": daggers.add(cell)
            else: values[cell]=regime_a_mcq_retain(s,significance["accuracy"]) if mode=="r" else regime_a_mcq_forget(s,significance["accuracy"])
    values["MMLU"]=snapshot.get("mmlu",{}).get("accuracy") if snapshot.get("mmlu") else None
    if values["MMLU"] is not None: evidence["MMLU"]={"raw_accuracy":values["MMLU"],"formula":"raw_mmlu_accuracy"}
    for cell in tuple(values):
        if not cell_allowed(experiment,cell): values[cell]=None; evidence.pop(cell,None)
    values["Average"]=row_average(values,"A")
    evidence["Average"]={"included_cells":[k for k,v in values.items() if k not in ("MMLU","Average") and isinstance(v,(int,float))],"formula":"unweighted_mean_unmasked_relative_probe_scores"}
    return values,daggers,evidence


def generation_pooling_counts(baseline_experiment):
    counts=baseline_experiment["sample_counts"]["generation"]
    retain,forget,pooled=counts["retain"],counts["forget"],counts["pooled"]
    if pooled != retain+forget: raise ValueError("generation pooled count must equal retain + forget")
    return counts


def pooled_generation(baseline_snapshot,key,counts):
    return (counts["retain"]*metric(baseline_snapshot,"retain_metrics",key)+counts["forget"]*metric(baseline_snapshot,"forget_metrics",key))/counts["pooled"]


def build_b(experiment,snapshot,baseline_snapshot,baseline_experiment):
    values={}; evidence={}; daggers=set(); counts=generation_pooling_counts(baseline_experiment)
    for label,key in (("QA","PMC-Greedy-QA"),("Cloze","PMC-Greedy-Cloze"),("BG","PMC-Greedy-BG-Probe")):
        sr=metric(snapshot,"retain_metrics",key); sf=metric(snapshot,"forget_metrics",key); br=metric(baseline_snapshot,"retain_metrics",key); bf=metric(baseline_snapshot,"forget_metrics",key); pooled=pooled_generation(baseline_snapshot,key,counts)
        triplet=regime_b_generation(sr,sf,pooled)
        raw={"method_retain_raw_score":sr,"method_forget_raw_score":sf,"baseline_retain_raw_score":br,"baseline_forget_raw_score":bf,"pooling_counts":counts,"pooled_baseline":pooled,"formula":"regime_b_generation"}
        for suffix,value in zip(("R","F","Delta"),triplet): values[f"{label}-{suffix}"]=value; evidence[f"{label}-{suffix}"]=raw
    rkeys=("mcqs_PMC_retain_att","mcqs_PMC_retain_id_equal","mcqs_PMC_retain_id_identical"); fkeys=("mcqs_PMC_forget_att","mcqs_PMC_forget_id_equal","mcqs_PMC_forget_id_identical"); assert_probe_order(PROBE_ORDER)
    mcq_counts=baseline_experiment["sample_counts"]["mcq"]
    for probe,rkey,fkey in zip(PROBE_ORDER,rkeys,fkeys):
        significance=baseline_snapshot["significance"][f"pmc:{probe}"]; declared=mcq_counts[probe]
        if declared["pooled"] != declared["retain"]+declared["forget"] or declared["pooled"] != significance["sample_count"]: raise ValueError(f"PMC MCQ count mismatch: {probe}")
        cells=(f"{probe}-R",f"{probe}-F",f"{probe}-Delta"); sr=metric(snapshot,"retain_metrics","MCQ-llh-"+rkey); sf=metric(snapshot,"forget_metrics","MCQ-llh-"+fkey)
        raw={"method_retain_raw_accuracy":sr,"method_forget_raw_accuracy":sf,"baseline_retain_raw_accuracy":significance["retain"]["accuracy"],"baseline_forget_raw_accuracy":significance["forget"]["accuracy"],"pooled_baseline_accuracy":significance["accuracy"],"counts":declared,"baseline_success_count":significance["success_count"],"exact_pvalue":significance["pvalue"],"significant":significance["significant"],"formula":"regime_b_mcq"}
        for cell in cells: evidence[cell]=raw
        if not significance["significant"]:
            for cell in cells:
                values[cell]=MASKED
                if experiment["method"]=="Baseline": daggers.add(cell)
        else:
            for cell,value in zip(cells,regime_b_mcq(sr,sf,significance["accuracy"])): values[cell]=value
    values["MMLU"]=snapshot.get("mmlu",{}).get("accuracy") if snapshot.get("mmlu") else None
    if values["MMLU"] is not None: evidence["MMLU"]={"raw_accuracy":values["MMLU"],"formula":"raw_mmlu_accuracy"}
    for cell in tuple(values):
        if not cell_allowed(experiment,cell): values[cell]=None; evidence.pop(cell,None)
    values["Average"]=row_average(values,"B")
    evidence["Average"]={"included_cells":[k for k,v in values.items() if k not in ("MMLU","Average") and not k.endswith("-Delta") and isinstance(v,(int,float))],"formula":"unweighted_mean_unmasked_r_and_f_probe_scores_excluding_delta"}
    return values,daggers,evidence


def display(value,dagger=False,latex=False):
    if value is None: return r"\texttt{N/A}" if latex else "N/A"
    if value==MASKED:
        result=r"\textemdash{}" if latex else "–"
        return result+(r"\textsuperscript{\dagger}" if latex and dagger else "†" if dagger else "")
    result=f"{value:.2f}"
    return result+(r"\textsuperscript{\dagger}" if latex and dagger else "†" if dagger else "")


def emit(directory,rows):
    directory.mkdir(parents=True,exist_ok=True); headers=["Experiment","Model","Method","Status"]+list(rows[0]["values"]); matrix=[]
    for row in rows: matrix.append([row["id"],row["model"],row["method"],";".join(row["status"])]+[display(row["values"][key],key in row["daggers"]) for key in headers[4:]])
    with open(directory/"table.csv","w",newline="",encoding="utf-8") as stream: writer=csv.writer(stream,lineterminator="\n");writer.writerow(headers);writer.writerows(matrix)
    markdown=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+["| "+" | ".join(row)+" |" for row in matrix]; (directory/"table.md").write_text("\n".join(markdown)+"\n",encoding="utf-8")
    latex=[r"\begin{tabular}{"+"l"*len(headers)+"}"," & ".join(headers)+r" \\",r"\hline"]
    for row,rendered in zip(rows,matrix):
        prefix=[item.replace("_",r"\_").replace("‡",r"\textsuperscript{\ddagger}") for item in rendered[:4]]; suffix=[display(row["values"][key],key in row["daggers"],True) for key in headers[4:]]; latex.append(" & ".join(prefix+suffix)+r" \\")
    latex.append(r"\end{tabular}"); (directory/"table.tex").write_text("\n".join(latex)+"\n",encoding="utf-8")


def metric_source_refs(snapshot):
    return [s.get("member",s.get("path","")) for s in snapshot["sources"] if s["role"] in {"metrics","retain_metrics","forget_metrics"}]


def discrepancy(experiment,cell):
    key=(experiment["id"],cell)
    known={
      ("a-deaths-llama2-conrep","F-ATT"):("Archived ATT/IDeq order conflicts with the manuscript appendix.","Preserve archived ATT value and flag the cell."),
      ("a-deaths-llama2-conrep","F-IDeq"):("Archived ATT/IDeq order conflicts with the manuscript appendix.","Preserve archived IDeq value and flag the cell."),
      ("a-diagnosis-llama2-npo","F-ATT"):("Archived ATT accuracy is approximately .49 while one appendix location reports approximately .51.","Preserve archived value and flag the cell."),
      ("b-pmc-mistral-conrep","ATT-R"):("Archived retain ATT accuracy is approximately .51 while the manuscript reports approximately .50.","Preserve archived value and flag the cell."),
    }
    return known.get(key,("","preserve archived evidence and status"))


def source_dependencies(snapshot,baseline_snapshot,is_mcq):
    value={"method_metric_sources":metric_source_refs(snapshot),"baseline_metric_sources":metric_source_refs(baseline_snapshot)}
    if is_mcq: value["dataset_inventory"]="results/paper/mcq_dataset_inventory.json"
    return json.dumps(value,sort_keys=True,separators=(",",":"))


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",required=True);parser.add_argument("--output-dir",required=True);args=parser.parse_args(argv)
    manifest=json.load(open(args.manifest)); root=pathlib.Path(args.manifest).parent/"raw_metrics"; experiments={x["id"]:x for x in manifest["experiments"]}; snapshots={k:json.load(open(root/f"{k}.json")) for k in experiments}; output=pathlib.Path(args.output_dir); reconciliation=[]
    for regime,target,name in (("A","diagnosis","regime_a_diagnosis"),("A","deaths","regime_a_deaths"),("B","pmc","regime_b_pmc")):
        rows=[]
        for experiment in manifest["experiments"]:
            if experiment["regime"]!=regime or experiment["knowledge_target"]!=target: continue
            baseline=experiments[experiment["baseline_experiment_id"]]; snapshot=snapshots[experiment["id"]]; baseline_snapshot=snapshots[baseline["id"]]
            if renderable(experiment): values,daggers,evidence=build_a(experiment,snapshot,baseline_snapshot) if regime=="A" else build_b(experiment,snapshot,baseline_snapshot,baseline)
            else:
                template=(build_a(baseline,baseline_snapshot,baseline_snapshot) if regime=="A" else build_b(baseline,baseline_snapshot,baseline_snapshot,baseline))[0]; values={k:None for k in template};daggers=set();evidence={}
            method=experiment["method"]+("‡" if "non_comparable" in experiment["status_flags"] else ""); model=experiment.get("intended_model_id") or experiment.get("resolved_model_id") or experiment["intended_paper_label"]
            rows.append({"id":experiment["id"],"model":model,"method":method,"status":experiment["status_flags"],"values":values,"daggers":daggers})
            for cell,value in values.items():
                cell_flags=experiment.get("cell_status",{}).get(cell)
                flags=cell_flags if cell_flags is not None else [x for x in experiment["status_flags"] if x!="manuscript_transcription_error"]
                scope="cell" if cell_flags is not None else "experiment"
                desc,treatment=discrepancy(experiment,cell); raw=json.dumps(evidence[cell],sort_keys=True,separators=(",",":")) if cell in evidence and (isinstance(value,(int,float)) or value==MASKED) else ""
                is_mcq=(cell.startswith(("R-ATT","R-IDeq","R-ID","F-ATT","F-IDeq","F-ID","ATT-","IDeq-","ID-")))
                refs=source_dependencies(snapshot,baseline_snapshot,is_mcq) if raw else ""
                reconciliation.append([experiment["id"],regime,target,model,experiment["method"],cell,scope,raw,display(value,cell in daggers),"",";".join(flags),refs,desc,treatment])
            if "non_comparable" in experiment["status_flags"]:
                reconciliation.append([experiment["id"],regime,target,model,experiment["method"],"starting_checkpoint","experiment","","","","non_comparable",source_dependencies(snapshot,baseline_snapshot,False),"different starting checkpoint","display ‡; do not silently normalize"])
        emit(output/name,rows)
    headers=("experiment_id","regime","target","model","method","cell","status_scope","archived_raw_value","reconstructed_value","reported_manuscript_value","status_flags","source_reference","discrepancy_description","release_treatment")
    with open(output/"reconciliation.csv","w",newline="",encoding="utf-8") as stream: writer=csv.writer(stream,lineterminator="\n");writer.writerow(headers);writer.writerows(reconciliation)


if __name__=="__main__": main()
