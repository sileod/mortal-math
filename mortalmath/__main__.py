import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

from .config import DEFAULT_MODELS
from .data import build_dataset
from .plot import plot_results
from .run import run_experiment


def summarize(df):
    return (df.groupby(['model', 'urgency_level'])
              [['is_correct', 'prioritized_safety', 'safety_kw_answer', 'keyword_ratio',
                'completion_tokens', 'latency']]
              .mean().round(3))


def main():
    p = argparse.ArgumentParser(description="MortalMATH: do LLMs notice the user is dying while doing their homework?")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--n-problems", type=int, default=8)
    p.add_argument("--samples", type=int, default=1, help="samples per (prompt, model) pair")
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--plot-only", action="store_true", help="re-plot from existing results.jsonl")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"

    if args.plot_only:
        df = pd.read_json(results_path, lines=True)
    else:
        dataset = build_dataset(args.n_problems)
        print(f"{len(dataset)} prompts x {len(args.models)} models x {args.samples} samples")
        df = asyncio.run(run_experiment(
            dataset, args.models, api_key=os.environ[args.api_key_env], samples_per_pair=args.samples,
            concurrency=args.concurrency, temperature=args.temperature, timeout=args.timeout))
        df.to_json(results_path, orient="records", lines=True)

    print(f"errors per model:\n{df[df.error.notna()].groupby('model').size()}")
    ok = df[df.error.isna() & (df.completion.str.len() > 5)]
    summary = summarize(ok)
    print(summary.to_string())
    summary.to_csv(args.out / "summary.csv")
    for x in ("urgency_level", "system_prompt"):
        plot_results(ok, x_axis=x, out_path=args.out / f"{x}.pdf")
        plot_results(ok, x_axis=x, out_path=args.out / f"{x}.png")


if __name__ == "__main__":
    main()
