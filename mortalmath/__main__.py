import argparse
import asyncio
from pathlib import Path

from . import config, leaderboard
from .store import entry_name, list_runs, load_runs, protocol, save_run


DATASET_COLUMNS = ["problem_id", "problem", "ground_truth", "raw_solution", "urgency_level", "scenario_id",
                   "system_prompt", "user_prompt"]


def retry_errors(args, kwargs):
    """Rerun only the failed calls of stored entries and merge them back."""
    import pandas as pd
    from .judge import judge
    from .run import run_experiment

    for m in list_runs(args.out):
        if m["protocol_id"] != protocol()["id"] or (args.models and m["entry"] not in args.models):
            continue
        df = pd.read_json(m["dir"] / "results.jsonl.gz", lines=True, compression="gzip")
        failed = df.error.notna()
        if not failed.any():
            continue
        print(f"{m['entry']}: retrying {failed.sum()} failed calls")
        new = asyncio.run(run_experiment(df.loc[failed, DATASET_COLUMNS], [m["model"]], **kwargs))
        new = asyncio.run(judge(new)).set_index(df.index[failed])
        df = pd.concat([df[~failed], new.assign(model=m["entry"])]).sort_index()
        print(f"{m['entry']}: {df.error.notna().sum()} errors left -> {save_run(df, m['entry'], args.out, m['reasoning_effort'])}")
    print(leaderboard.write(args.out))


def cmd_run(args):
    from .data import build_dataset
    from .judge import judge
    from .run import run_experiment

    if args.retry_errors:
        return retry_errors(args, dict(max_concurrency=args.concurrency, temperature=config.TEMPERATURE,
                                       max_tokens=config.MAX_TOKENS, timeout=args.timeout, show_progress=False))
    pid = protocol()["id"]
    done = {m["entry"] for m in list_runs(args.out) if m["protocol_id"] == pid}
    entries = {entry_name(m, args.reasoning_effort): m for m in args.models}
    todo = {e: m for e, m in entries.items() if args.force or args.estimate or e not in done}
    for e in entries.keys() - todo.keys():
        print(f"skip {e}: already in {args.out}/runs under protocol {pid} (use --force to rerun)")
    if not todo:
        return

    dataset = build_dataset(1 if args.estimate else config.N_PROBLEMS)
    kwargs = dict(max_concurrency=args.concurrency, temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
                  timeout=args.timeout, show_progress=False)
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort
    print(f"{len(dataset)} prompts x {len(todo)} models, protocol {pid}")

    df = asyncio.run(run_experiment(dataset, list(todo.values()), **kwargs))
    df = asyncio.run(judge(df))
    df["model"] = df.model.map({m: e for e, m in todo.items()})

    for e in todo:
        d = df[df.model == e]
        cost = d.cost.sum() + d.judge_cost.sum()
        if args.estimate:
            print(f"{e}: ${cost:.3f} for {len(d)} prompts -> about ${cost * config.N_PROBLEMS:.2f} for a full run")
        else:
            print(f"{e}: ${cost:.2f}, {d.error.notna().sum()} errors -> {save_run(d, e, args.out, args.reasoning_effort)}")
    if not args.estimate:
        print(leaderboard.write(args.out))


def cmd_leaderboard(args):
    print(leaderboard.write(args.out))


def cmd_report(args):
    from .report import write_report
    print(write_report(args.out, args.focus, args.findings, args.highlight))


def cmd_plot(args):
    from .plot import plot_results
    df = load_runs(args.out)
    df = df[df.error.isna()]
    for x in ("urgency_level", "system_prompt"):
        for ext in ("pdf", "png"):
            plot_results(df, x_axis=x, out_path=args.out / f"{x}.{ext}")


def main():
    p = argparse.ArgumentParser(prog="mortalmath", description="Do LLMs notice the user is in danger while doing their homework?")
    p.add_argument("--out", type=Path, default=Path("results"), help="results root (runs/, leaderboard, report)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate models, judge replies, store runs, update the leaderboard")
    r.add_argument("models", nargs="*", default=config.DEFAULT_MODELS, help="OpenRouter model ids")
    r.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], help="stored as its own entry, e.g. model@low")
    r.add_argument("--estimate", action="store_true", help="run 1 problem and print the projected cost of a full run; stores nothing")
    r.add_argument("--force", action="store_true", help="rerun entries already stored under the current protocol")
    r.add_argument("--retry-errors", action="store_true", help="rerun only failed calls of stored entries (all, or the given ids)")
    r.add_argument("--concurrency", type=int, default=128, help="max concurrent requests per model")
    r.add_argument("--timeout", type=float, default=120)
    r.set_defaults(fn=cmd_run)

    sub.add_parser("leaderboard", help="recompute leaderboard.{json,csv} and the README table").set_defaults(fn=cmd_leaderboard)

    rep = sub.add_parser("report", help="build <out>/report.html")
    rep.add_argument("--focus", default=config.DEFAULT_MODELS[0], help="entry highlighted in the report")
    rep.add_argument("--findings", type=Path, help="text file, one finding per line (inline HTML allowed)")
    rep.add_argument("--highlight", nargs="*", default=config.DEFAULT_MODELS[1:], help="comparison entries drawn in color")
    rep.set_defaults(fn=cmd_report)

    sub.add_parser("plot", help="matplotlib figures (urgency_level / system_prompt)").set_defaults(fn=cmd_plot)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
