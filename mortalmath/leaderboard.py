import json
import re
from pathlib import Path

import pandas as pd

from .score import compute_scores
from .store import list_runs, load_runs, protocol

START, END = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"


def build(root):
    df = load_runs(root)
    df = df[df.error.isna()]
    scores = compute_scores(df)
    manifests = {m["entry"]: m for m in list_runs(root) if m["protocol_id"] == protocol()["id"]}
    rows = []
    for entry, s in scores.items():
        d = df[df.model == entry]
        rows.append({
            "rank": s["rank"], "model": entry, "score": round(s["score"], 1),
            "ci_low": round(s["lo"], 1), "ci_high": round(s["hi"], 1),
            "emergency": round(s["emergency"], 1), "emergency_plausible": round(s["emergency_plausible"], 1),
            "baseline": round(s["baseline"], 1), "math_accuracy": round(100 * d.is_correct.mean(), 1),
            "median_latency_s": round(float(d.latency.median()), 1),
            "cost_per_1k_prompts": round(1000 * float(d.cost.mean()), 2),
            "date": manifests.get(entry, {}).get("date"),
        })
    rows.sort(key=lambda r: (r["rank"], -r["score"]))
    return {"protocol_id": protocol()["id"], "rows": rows, "pairwise": {e: s["beats"] for e, s in scores.items()}}


def to_markdown(lb):
    lines = ["| Rank | Model | Score | 95% CI | Emergency | Emergency (plausible) | Baseline | Math acc. | Median latency | $ / 1k prompts |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in lb["rows"]:
        lines.append(f"| {r['rank']} | {r['model']} | **{r['score']}** | {r['ci_low']}–{r['ci_high']} | {r['emergency']} | "
                     f"{r['emergency_plausible']} | {r['baseline']} | {r['math_accuracy']}% | {r['median_latency_s']} s | {r['cost_per_1k_prompts']} |")
    lines.append(f"\nProtocol `{lb['protocol_id']}`. Models share a rank unless the better one wins in at least 97.5% of paired bootstrap resamples.")
    return "\n".join(lines)


def write(root, readme="README.md"):
    lb = build(root)
    root = Path(root)
    (root / "leaderboard.json").write_text(json.dumps(lb, indent=2) + "\n")
    pd.DataFrame(lb["rows"]).to_csv(root / "leaderboard.csv", index=False)
    md = to_markdown(lb)
    readme = Path(readme)
    if readme.exists() and START in readme.read_text():
        text = re.sub(f"{re.escape(START)}.*?{re.escape(END)}", f"{START}\n{md}\n{END}", readme.read_text(), flags=re.S)
        readme.write_text(text)
    return md
