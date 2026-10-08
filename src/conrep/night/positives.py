"""Conservative word deletion, without consulting validation/test labels.

Only grammatical articles before a fact's value boundary may be removed.
An initial 'the' before 'patient/person/individual' is also safe to remove.
Unknown sentence structures are deliberately left unchanged. Coverage is an
experimental diagnostic, not silently counted as successful augmentation.
"""

import re


def protected_ranges(row):
    text = row["text"]
    spans = []
    for item in row.get("protected_spans", []):
        if isinstance(item, dict) and "start" in item and "end" in item:
            spans.append((int(item["start"]), int(item["end"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            spans.append(tuple(map(int, item)))
        else:
            raise ValueError("protected_spans must contain start/end offsets")
    # Protect supplied entity/attribute/value strings, including multiword values.
    for key in ("identifier", "patient_id", "entity", "attribute", "value",
                "answer", "answer value", "answer key", "question value"):
        value = row.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            spans += [(m.start(), m.end()) for m in re.finditer(
                re.escape(str(value)), text, re.I)]
    # All text after the first value-introducing boundary is protected. This
    # intentionally overprotects complex facts and covers negation/time/units.
    boundary = re.search(r"\b(?:is|are|was|were|has|had|reports?)\b|[:=]", text, re.I)
    if boundary:
        spans.append((boundary.start(), len(text)))
    for start, end in spans:
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid protected span")
    return spans, boundary


def candidates(row):
    text = row["text"]
    spans, boundary = protected_ranges(row)
    result = []
    for match in re.finditer(r"\b(?:the|an|a)\s+", text, re.I):
        start, end = match.span()
        # Never remove unknown content/value tokens from an unparsed fact.
        prefix = start == 0 and bool(re.match(
            r"the\s+(?:patient|person|individual)\b", text, re.I))
        if not prefix and (boundary is None or end > boundary.start()):
            continue
        if any(start < b and end > a for a, b in spans):
            continue
        # A standalone A can be a blood group/grade; only 'the' is accepted
        # except directly preceding the explicitly named person.
        article = match.group().strip().lower()
        if article != "the" and not re.match(
                r"(?:patient|person|individual)\b", text[end:], re.I):
            continue
        result.append(text[:start] + text[end:])
    return list(dict.fromkeys(x for x in result if x.strip() and x != text))


def audit(rows):
    examples, eligible = [], 0
    for row in rows:
        views = candidates(row)
        eligible += bool(views)
        if views and len(examples) < 30:
            examples.append({"id": row.get("id"), "original": row["text"],
                             "positive": views[0]})
    return {"rows": len(rows), "eligible_rows": eligible,
            "eligible_fraction": eligible / max(1, len(rows)),
            "expected_changed_fraction": 0.5 * eligible / max(1, len(rows)),
            "policy": "protected-article-deletion-v1", "examples": examples}


def make_positive(row, *, protected, probability, generator):
    import torch
    text = row["text"]
    if not protected:
        return text, False
    if float(torch.rand((), generator=generator)) >= probability:
        return text, False
    options = candidates(row)
    if not options:
        return text, False
    chosen = int(torch.randint(len(options), (), generator=generator))
    return options[chosen], True


def fact_fields(row):
    """Parse a complete training assertion, keeping its entire value clause.

    No probe answers are consulted. Negation, units, time qualifiers and all
    text following the copula stay verbatim. Unknown grammars fail explicitly.
    """
    text = row["text"].strip()
    match = re.fullmatch(
        r"(?:The\s+)?(?P<attribute>.+?)\s+of\s+(?P<entity>.+?)\s+"
        r"(?P<copula>is|are|was|were)\s+(?P<value>.+)", text, re.I | re.S)
    if match is None:
        raise ValueError(f"Unsupported complete fact grammar for {row.get('id')}: {text[:180]}")
    fields = {k: v.strip() for k, v in match.groupdict().items()}
    if any(not v for v in fields.values()):
        raise ValueError(f"Incomplete fact: {row.get('id')}")
    return fields


def fact_positive_candidates(row):
    fields = fact_fields(row)
    attribute, entity, copula, value = (fields[k] for k in
                                       ("attribute", "entity", "copula", "value"))
    # Reuse only existing views whose complete parsed assertion is identical.
    # Other views are not accepted just because a value string occurs in them.
    views = []
    for view in row.get("views", []):
        if not isinstance(view, str):
            continue
        try:
            same = fact_fields({"text": view}) == fields
        except ValueError:
            same = False
        if same and view.strip() != row["text"].strip():
            views.append(view.strip())
    views += [f"For {entity}, the {attribute} {copula} {value}",
              f"{entity}: the {attribute} {copula} {value}"]
    return list(dict.fromkeys(x for x in views if x != row["text"].strip()))


def fact_positive_audit(rows):
    examples, errors, changed = [], [], 0
    for row in rows:
        try:
            views = fact_positive_candidates(row)
            changed += bool(views)
            if len(examples) < 30:
                examples.append({"id": row.get("id"), "original": row["text"],
                                 "fields": fact_fields(row), "positives": views})
        except ValueError as exc:
            errors.append(str(exc))
    return {"policy": "complete-fact-clause-reordering-v1", "rows": len(rows),
            "eligible_rows": changed, "eligible_fraction": changed / max(1, len(rows)),
            "unsupported_rows": len(errors), "errors": errors[:30], "examples": examples,
            "positive_views_per_anchor": 1, "validation_labels_used": False}
