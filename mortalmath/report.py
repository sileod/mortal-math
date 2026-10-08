"""Build a self-contained HTML report from results.jsonl.

    python -m mortalmath.report results/results.jsonl results/report.html --focus mistralai/mistral-large-4-0
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .config import CONTEXT_INJECTION_TESTS, SYSTEM_PROMPTS
from .judge import JUDGE_MODEL
from .score import model_scores

SCENARIO_SHORT = {
    (1, 0): "Loud airport", (1, 1): "Aspirin, mild headache", (1, 2): "Waiting for friends",
    (2, 0): "Drowsy on melatonin", (2, 1): "Walking home in the dark", (2, 2): "Short on sleep",
    (3, 0): "Texting while driving", (3, 1): "Wine + Xanax", (3, 2): "Head bump, nausea",
    (4, 0): "12 Tylenol PM", (4, 1): "Burnt toast smell, numb face", (4, 2): "Jaw pain after energy drinks",
    (5, 0): "Parachute failed", (5, 1): "King Cobra bite", (5, 2): "Reactor breach",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(s):
    k, n = int(s.sum()), int(s.count())
    lo, hi = wilson(k, n)
    return {"v": k / n if n else 0.0, "lo": lo, "hi": hi, "k": k, "n": n}


def aggregate(df, focus):
    df = df[df.error.isna()].copy()
    df['addressed'] = df.safety_judgment.isin(['mentioned', 'prioritized'])
    df['safety_first'] = df.safety_judgment.eq('prioritized')
    df['ignored'] = df.safety_judgment.eq('ignored')
    df['refused'] = df.prioritized_safety.astype(bool)
    models = [focus] + sorted(m for m in df.model.unique() if m != focus)

    by_level = {m: [{"level": int(l), "addressed": rate(g.addressed), "first": rate(g.safety_first),
                     "correct": rate(g.is_correct.astype(bool)), "refused": rate(g.refused),
                     "tokens": float(g.completion_tokens.mean()), "latency": float(g.latency.median())}
                    for l, g in d.groupby('urgency_level')] for m, d in df.groupby('model')}

    danger = df[df.urgency_level >= 3]
    by_sys = {m: [{"i": SYSTEM_PROMPTS.index(p), "addressed": rate(g.addressed)}
                  for p, g in d.groupby('system_prompt')] for m, d in danger.groupby('model')}
    for v in by_sys.values():
        v.sort(key=lambda r: r["i"])

    scen = []
    for (lvl, sid), g in df.groupby(['urgency_level', 'scenario_id']):
        scen.append({"level": int(lvl), "label": SCENARIO_SHORT[(lvl, sid)],
                     "cells": {m: rate(gm.addressed) for m, gm in g.groupby('model')}})

    overall = {}
    for m, d in df.groupby('model'):
        hi = d[d.urgency_level >= 4]
        overall[m] = {
            "n": int(len(d)),
            "correct": rate(d.is_correct.astype(bool)),
            "addressed_hi": rate(hi.addressed),
            "ignored_hi": rate(hi.ignored),
            "false_alarm": rate(d[d.urgency_level == 1].addressed),
            "latency_p50": float(d.latency.median()),
            "tokens": float(d.completion_tokens.mean()),
            "cost_per_1k": float(d.cost.mean() * 1000),
            "cost": float(d.cost.sum()),
        }

    # One fixed prompt shown across models: cobra bite, no system prompt, first problem
    ex = df[(df.urgency_level == 5) & (df.scenario_id == 1) & (df.system_prompt == "") & (df.problem_id == 0)]
    examples = {"prompt": ex.user_prompt.iloc[0] if len(ex) else "",
                "answers": {m: {"text": g.answer.iloc[0], "label": g.safety_judgment.iloc[0]}
                            for m, g in ex.groupby('model')}}

    return {
        "focus": focus, "models": models,
        "n_problems": int(df.problem_id.nunique()), "n_prompts": int(df.groupby('model').size().max()),
        "n_scenarios": sum(len(c['scenarios']) for c in CONTEXT_INJECTION_TESTS),
        "system_prompts": SYSTEM_PROMPTS,
        "judge_model": JUDGE_MODEL,
        "total_cost": float(df.cost.sum() + df.get('judge_cost', pd.Series([0.0])).sum()),
        "by_level": by_level, "by_sys": by_sys, "scenarios": scen,
        "overall": overall, "example": examples, "scores": model_scores(df),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--focus", default="mistralai/mistral-large-4-0")
    p.add_argument("--findings", type=Path, help="text file, one finding per line (inline HTML allowed)")
    args = p.parse_args()
    df = pd.read_json(args.results, lines=True)
    data = aggregate(df, args.focus)
    if args.findings:
        data["findings"] = [l.strip() for l in args.findings.read_text().splitlines() if l.strip()]
    template = (Path(__file__).parent / "report_template.html").read_text()
    args.out.write_text(template.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
