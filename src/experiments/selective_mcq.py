"""Binary counterpart of frozen direct-choice-v1; four-option scoring stays native.

No gold answer is passed to extraction. Out-of-domain C/D labels abstain.
"""
import re
from .answer_extraction import surface, canonical, negative, LEADING, REVERSE, DECLARATIONS, HEDGE

VERSION = "direct-choice-binary-v1"


def extract_binary(output, options):
    if set(options) != set("AB") or any(not str(x).strip() for x in options.values()):
        raise ValueError("Two nonempty options A/B required")
    text = surface(output)
    def result(label, status, evidence=""):
        return dict(predicted=label, status=status, evidence=evidence, parser=VERSION)
    if not text:
        return result(None, "empty")
    # Ignore an exact echo of all supplied options, never an arbitrary labelled line.
    echoed = list(re.finditer(r"(?m)^\s*([A-D])[.)]\s+(.+)$", text))
    if [m[1] for m in echoed] == list("AB") and all(canonical(m[2]) == canonical(options[m[1]]) for m in echoed):
        text = text[:echoed[0].start()] + text[echoed[-1].end():]
        text = text.strip()
        if not text:
            return result(None, "option_echo_only")
    candidates = []
    matches = [(m, "leading") for m in [LEADING.match(text)] if m]
    matches += [(m, "reverse") for m in REVERSE.finditer(text)]
    matches += [(m, "declaration") for p in DECLARATIONS for m in p.finditer(text)]
    # A labelled answer line is accepted only when its attached text verifies it.
    lines = list(re.finditer(r"(?m)^\s*([A-D])[.)]\s+(.+)$", text))
    for m in lines if len(lines) == 1 else []:
        if m[1] in options and canonical(m[2]) == canonical(options[m[1]]):
            candidates.append((m[1], m[0], m.end(), "verified_labelled_line"))
    for match, kind in matches:
        label = next(x for x in match.groups() if x).upper()
        tail = text[match.end():]
        # 'answer is a firm mass', including sentence-initial uppercase A.
        if kind == "declaration" and re.match(r"\s+[A-Za-z]", tail) and not re.match(r"\s+(?:because|or|and)\b", tail, re.I):
            continue
        if negative(text, match.start()) or re.search(r"\b(?:example|says|said)\s*[:\"']?\s*$", text[:match.start()], re.I):
            continue
        if kind != "reverse" and re.match(r"\s*(?:or|and|/)\s*[A-D]\b", tail, re.I):
            return result(None, "ambiguous_labels", match[0] + tail[:25])
        candidates.append((label, match[0], match.end(), kind))
    # 'The false statement is D. <full option>' and similar answer clauses.
    for m in re.finditer(r"\b(?:is|be|by)\s+([A-D])\.\s+([^\n]+)", text):
        if m[1] not in options:
            continue
        value = canonical(options[m[1]])
        tail = canonical(m[2])
        if value and (tail == value or tail.startswith(value + ". ")) and not negative(text, m.start()):
            candidates.append((m[1], m[0], m.end(), "verified_labelled_content"))
    for m in re.finditer(r"\b(?:answer|translation|symbolization|statement|proposition|benefit|class of model)[^\n]{0,180}?\s(?:is|be)\s([A-D])\.", text):
        if not negative(text, m.start()) and not re.search(r"\b(?:example|says|said)\s*[:\"']?\s*$", text[:m.start()], re.I):
            candidates.append((m[1], m[0], m.end(), "answer_clause_label"))
    # A trailing standalone letter after reasoning is still an explicit choice.
    last = re.search(r"\n\s*([A-D])\s*$", text)
    if last:
        candidates.append((last[1], last[0], last.end(), "final_line"))
    labels = {x[0] for x in candidates}
    if len(labels) > 1:
        return result(None, "conflicting_labels", " | ".join(x[1] for x in candidates))
    if labels:
        label = next(iter(labels))
        if label not in options:
            return result(None, "invalid_label", candidates[0][1])
        denial = (re.search(r"\b(?:not|neither|isn't)\s+(?:option|choice)\s+" + label + r"\b", text, re.I)
                  or re.search(r"\b" + label + r"\s+is\s+(?:not\s+(?:correct|the answer)|wrong|incorrect)\b", text, re.I))
        denial = denial or re.search(r"\b(?:not|Not|neither|Neither)\s+" + label + r"(?=$|[.,;!?]|\s+because\b)", text)
        revision = re.search(r"\b(?:actually|instead|rather|correction)\s*[:,]?\s*[A-D]\b", text, re.I)
        if denial or revision or HEDGE.search(text[:min(x[2] for x in candidates)]):
            return result(None, "qualified_choice", candidates[0][1])
        return result(label, "explicit_label", candidates[0][1])
    if HEDGE.search(text):
        return result(None, "uncertain_or_refusal")
    normal_options = {k: canonical(v) for k, v in options.items()}
    # Identical option text has no unique label. Explicit labels above are valid.
    def resolve(piece, evidence):
        keys = [k for k, v in normal_options.items() if canonical(piece) == v]
        if len(keys) > 1: return result(None, "duplicate_option_content", evidence)
        return result(keys[0], "direct_option_content", evidence) if keys else None
    whole = resolve(text, text)
    if whole: return whole
    # Full answer-clause equality is safe even for very short numeric choices.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text)
    marked = [s for s in sentences if re.match(r"(?:Therefore|Thus|Hence|So,|In total,|The (?:final |correct )?answer\b|The closest (?:answer|option)\b)", s)]
    clauses = list(dict.fromkeys([sentences[0], sentences[-1], *marked]))
    scalar = all(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", v) for v in normal_options.values())
    contents = []
    for clause in clauses:
        direct = []
        found = resolve(clause, clause)
        if found: direct.append(found)
        for m in re.finditer(r"\b(?:is|are|was|were|be|equals|equal to|called|contains|refers to|given by the formula|bound to)\s+|=\s*", clause, re.I):
            if negative(clause, m.start()): continue
            piece = clause[m.end():]
            piece = re.split(r",?\s+(?:because|since|as indicated|which (?:is|are)|as it|as they|ignoring|according to)\b", piece, maxsplit=1, flags=re.I)[0]
            if scalar:
                piece = re.sub(r"\s+(?:edges|days|parameters|lamps|seconds|units)\.?$", "", piece)
            found = resolve(piece, clause)
            if found: direct.append(found)
        if direct:
            contents.extend(direct)
            continue
        # Full multiword content, with a competing-option check in the same
        # clause. No partial numeric match, stemming or embedding similarity.
        c = canonical(clause)
        hits = []
        for label, value in normal_options.items():
            if len(re.findall(r"\w+", value)) < 2 or re.search(r"\d", value): continue
            for m in re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", c):
                if not negative(c, m.start()): hits.append((label, m))
        if len({x[0] for x in hits}) > 1:
            contents.append(result(None, "ambiguous_content", clause))
        elif hits:
            label, match = hits[0]
            if not re.match(r"\s*(?:\([^)]*\)\s*)?(?:or|and|/|,)\s+", c[match.end():], re.I):
                contents.append(result(label, "direct_option_content", clause))
    # Prefer a uniquely specified explicit final conclusion over intermediate
    # arithmetic, otherwise do not resolve conflicting content guesses.
    final = [x for x in contents if x["evidence"] in marked]
    if final:
        contents = final
    if any(x["predicted"] is None for x in contents):
        return result(None, "ambiguous_content")
    labels = {x["predicted"] for x in contents}
    if len(labels) == 1:
        return contents[0]
    return result(None, "ambiguous_content" if labels else "unparsed")

