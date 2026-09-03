import re
from utils import answer_present

def greedy_hit(generated, ans): return int(answer_present(generated, ans))

def parse_letter(text):
    m=re.search(r"\b([A-Z])\b",text); 
    return m.group(1) if m else None

def mcq_hit(output, correct_letter): return int(parse_letter(output)==correct_letter)

