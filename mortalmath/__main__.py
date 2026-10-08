import argparse
import asyncio
from pathlib import Path

import pandas as pd

from .config import DEFAULT_MODELS
from .data import build_dataset
from .judge import judge
from .plot import plot_results
from .run import run_experiment
from .score import model_scores


def summarize(df):
    cols = ['is_correct', 'prioritized_safety', 'safety_kw_answer', 'keyword_ratio', 'completion_tokens', 'latency']
    if 'safety_judgment' in df:
        df = df.assign(safety_addressed=df.safety_judgment.isin(['mentioned', 'prioritized']),
                       safety_first=df.safety_judgment.eq('prioritized'))
        cols = ['safety_addressed', 'safety_first'] + cols
    return df.groupby(['model', 'urgency_level'])[cols].mean().round(3)


def main():
    p = argparse.ArgumentParser(description="MortalMATH: do LLMs notice the user is dying while doing their homework?")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--n-problems", type=int, default=8)
    p.add_argument("--samples", type=int, default=1, help="samples per (prompt, model) pair")
    p.add_argument("--concurrency", type=int, default=128, help="max concurrency per model")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--plot-only", action="store_true", help="re-plot from existing results.jsonl")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM safety judge")
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"

    if args.plot_only:
        df = pd.read_json(results_path, lines=True)
    else:
        dataset = build_dataset(args.n_problems)
        print(f"{len(dataset)} prompts x {len(args.models)} models x {args.samples} samples")
        df = asyncio.run(run_experiment(
            dataset, args.models, samples_per_pair=args.samples, max_concurrency=args.concurrency,
            temperature=args.temperature, timeout=args.timeout, max_tokens=args.max_tokens,
            show_progress=False))
        print(f"total cost: ${df.cost.sum():.3f}")
        df.to_json(results_path, orient="records", lines=True)

    if not args.no_judge and 'safety_judgment' not in df:
        df = asyncio.run(judge(df))
        print(f"judge cost: ${df.judge_cost.sum():.3f}")
        df.to_json(results_path, orient="records", lines=True)

    print(f"errors per model:\n{df[df.error.notna()].groupby('model').size()}")
    ok = df[df.error.isna() & (df.completion.str.len() > 5)]
    summary = summarize(ok)
    print(summary.to_string())
    summary.to_csv(args.out / "summary.csv")
    if 'safety_judgment' in ok:
        scores = pd.DataFrame(model_scores(ok)).T.sort_values('score', ascending=False).round(1)
        print(f"\nMortalMATH score:\n{scores.to_string()}")
        scores.to_csv(args.out / "scores.csv")
    for x in ("urgency_level", "system_prompt"):
        plot_results(ok, x_axis=x, out_path=args.out / f"{x}.pdf")
        plot_results(ok, x_axis=x, out_path=args.out / f"{x}.png")


if __name__ == "__main__":
    main()
