import yaml, argparse, logging
import os
from utils import load_jsonl, set_seed, ensure_dir
from model_loader import load_model
from mcq_loader import load_all_mcq
from evaluator import Evaluator

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    cfg=yaml.safe_load(open(parser.parse_args().config))
    ensure_dir(cfg["logging"]["out_dir"])
    logging.basicConfig(level=logging.INFO,
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(cfg["logging"]["out_dir"],"run.log"),"w",encoding="utf-8")])
    set_seed(cfg["seed"])

    # Load model
    tokenizer, model = load_model(cfg)

    # QA datasets
    deaths_rows   = load_jsonl(cfg["dataset"]["qa_files"]["deaths"])
    diag_rows     = load_jsonl(cfg["dataset"]["qa_files"]["diagnosis"])
    for r in deaths_rows: r["__domain"]="celebrity_deaths"
    for r in diag_rows:   r["__domain"]="celebrity_diagnosis"

    # MCQ datasets
    mcq_rows = load_all_mcq(cfg["dataset"]["mcq_files"])

    logging.info(f"Loaded QA rows: deaths={len(deaths_rows)}, diagnosis={len(diag_rows)}")
    logging.info(f"Loaded MCQ rows: {len(mcq_rows)}")

    evaluator = Evaluator(cfg, tokenizer, model, deaths_rows, diag_rows, mcq_rows)
    summary = evaluator.run()

    logging.info("=== SUMMARY ===")
    for k,v in summary.items():
        logging.info(f"{k:<35} acc={v['acc']:.3f} "
                     f"{'prob='+str(v.get('prob',''))} "
                     f"{'topk='+str(v.get('topk_leak',''))} "
                     f"n={v['n']}")

if __name__=="__main__":
    main()

