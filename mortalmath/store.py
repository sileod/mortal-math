"""One directory per leaderboard entry: results/runs/<slug>/{results.jsonl.gz, manifest.json}."""
import datetime as dt
import hashlib
import inspect
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from . import config, judge, score
from .data import construct_prompt


def entry_name(model, reasoning_effort=None):
    return f"{model}@{reasoning_effort}" if reasoning_effort else model


def slug(entry):
    return entry.replace("/", "__").replace("@", "--")


def protocol():
    """Settings that must match for two runs to be comparable, plus a short hash of them."""
    p = {
        "dataset": config.DATASET, "dataset_revision": config.DATASET_REVISION,
        "n_problems": config.N_PROBLEMS, "temperature": config.TEMPERATURE, "max_tokens": config.MAX_TOKENS,
        "judge_model": config.JUDGE_MODEL,
        "system_prompts": config.SYSTEM_PROMPTS, "scenarios": config.CONTEXT_INJECTION_TESTS,
        "prompt_template": inspect.getsource(construct_prompt), "judge_template": judge.TEMPLATE,
        "credit": score.CREDIT,
    }
    p["id"] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]
    return p


def _pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except OSError:
        return None


def save_run(df, entry, root, reasoning_effort=None):
    d = Path(root) / "runs" / slug(entry)
    d.mkdir(parents=True, exist_ok=True)
    df.assign(model=entry).to_json(d / "results.jsonl.gz", orient="records", lines=True, compression="gzip")
    manifest = {
        "entry": entry, "model": entry.split("@")[0], "reasoning_effort": reasoning_effort,
        "date": dt.date.today().isoformat(), "protocol_id": protocol()["id"],
        "n_prompts": int(len(df)), "n_errors": int(df.error.notna().sum()),
        "cost_usd": round(float(df.cost.sum()), 4), "judge_cost_usd": round(float(df.get("judge_cost", pd.Series([0])).sum()), 4),
        "git_commit": _git_commit(),
        "versions": {k: _pkg(k) for k in ("litlm", "litellm", "math-verify", "datasets")},
        "protocol": {k: v for k, v in protocol().items() if k in ("dataset", "dataset_revision", "n_problems", "temperature", "max_tokens", "judge_model")},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return d


def list_runs(root):
    return [json.loads(p.read_text()) | {"dir": p.parent} for p in sorted(Path(root).glob("runs/*/manifest.json"))]


MAX_ERROR_RATE = 0.05


def load_runs(root, protocol_id=None, entries=None, max_error_rate=MAX_ERROR_RATE):
    """Concatenate stored runs under the current protocol, skipping incomplete ones (too many failed calls)."""
    protocol_id = protocol_id or protocol()["id"]
    frames = []
    for m in list_runs(root):
        if m["protocol_id"] != protocol_id or (entries and m["entry"] not in entries):
            continue
        if m["n_errors"] > max_error_rate * m["n_prompts"]:
            print(f"skip {m['entry']}: {m['n_errors']}/{m['n_prompts']} failed calls; "
                  f"run `python -m mortalmath run {m['entry'].split('@')[0]} --retry-errors`")
            continue
        frames.append(pd.read_json(m["dir"] / "results.jsonl.gz", lines=True, compression="gzip"))
    if not frames:
        raise SystemExit(f"no runs under protocol {protocol_id} in {root}/runs")
    return pd.concat(frames, ignore_index=True)
