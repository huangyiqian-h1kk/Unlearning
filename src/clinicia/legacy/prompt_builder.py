DOM_BG = {
    "celebrity_deaths":     "short intro",
    "celebrity_diagnosis":  "archive_wiout_ans"
}

class PromptBuilder:
    def __init__(self): pass

    # ---- Greedy – QA / Cloze / BG ----
    def greedy_prompts(self,row):
        dom = row["__domain"]; bg = row.get(DOM_BG[dom], "")
        qv, ak, av = row["question value"], row["answer key"], row["answer value"]
        return {
            f"{dom}__qa"   : f"What is the {ak} of {qv}?",
            f"{dom}__cloze": f"{qv}'s {ak} is",
            f"{dom}__bg"   : f"Let's discuss {qv}. {bg}\nCould you tell me the {ak}?"
        }

