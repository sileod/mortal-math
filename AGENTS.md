# AGENTS.md

A guide for coding agents working on MortalMATH: adding models to the leaderboard, changing the score or the
protocol, and building the report.

## Layout

| Path | Role |
|---|---|
| `mortalmath/config.py` | **Protocol**: dataset revision, n problems, temperature, token cap, judge model, scenarios, system prompts |
| `mortalmath/data.py` | Builds the 3,000 prompts (problems × scenarios × system prompts) |
| `mortalmath/run.py` | Model calls through `litlm.acomplete` (all models in parallel), per-reply metrics |
| `mortalmath/judge.py` | LLM judge: `ignored` / `mentioned` / `prioritized` |
| `mortalmath/score.py` | The MortalMATH score, bootstrap CIs, ranks with ties, pairwise win rates |
| `mortalmath/store.py` | Run storage, manifests, protocol hash |
| `mortalmath/leaderboard.py` | `results/leaderboard.{json,csv}` and the README table between the `leaderboard` markers |
| `mortalmath/report.py`, `report_template.html` | Self-contained HTML report |
| `results/runs/<entry>/` | One leaderboard entry: `results.jsonl.gz` + `manifest.json` |
| `tests/` | Offline tests (no API calls) |

## Add a model to the leaderboard

1. Check the id exists: `curl -s https://openrouter.ai/api/v1/models | jq -r '.data[].id' | grep <name>`.
2. **Estimate cost first**: `python -m mortalmath run <id> --estimate`. A full run is 3,000 prompts plus 3,000
   judge calls. Reasoning models can cost 10–25× more than non-reasoning ones (Mistral Large 4: about $10;
   Haiku 5.5: about $0.75). Tell the user the estimate before a run that costs more than a few dollars.
3. Run: `python -m mortalmath run <id>` (several ids run in parallel). This stores the run and rewrites the leaderboard.
4. Check `n_errors` in the run's `manifest.json`. Failed calls (usually upstream 429 rate limits) are stored with
   `error` set. Entries with more than 5% failed calls are left off the leaderboard until fixed:
   `python -m mortalmath run <id> --retry-errors --concurrency 6` reruns only the failed calls. Repeat it if needed.
5. Commit `results/runs/<entry>/`, `results/leaderboard.*` and `README.md`. Rebuild the report with
   `python -m mortalmath report --focus <entry> --findings results/findings.txt` and update the findings to match.

Use `--reasoning-effort low|medium|high` to add a variant as its own entry (`<id>@low`).

## Rules

- **Never commit API keys.** Keys come from the environment (`OPENROUTER_API_KEY`). Don't echo them.
- **The judge must not be a model under evaluation.** It is fixed in `config.JUDGE_MODEL` so that all entries are judged the same way.
- **Changing the protocol invalidates the leaderboard.** Any edit to `config.py` settings, the prompt or judge
  templates, or `score.CREDIT` changes the protocol id, and old runs drop off the leaderboard until rerun.
  Do it only on purpose, say so in the commit message, and rerun every entry.
- **Score changes that keep the protocol** (aggregation, bootstrap, weighting in `score.py` outside `CREDIT`)
  are recomputed from stored runs with `python -m mortalmath leaderboard`; no API calls needed. Keep `tests/test_score.py`
  passing, and add a test for any new property (for example "always panicking scores 0").
- Rerunning an entry under the same protocol needs `--force`. Prefer adding entries to overwriting them.
- `results/runs/*/results.jsonl.gz` holds full replies; read it with
  `pd.read_json(path, lines=True, compression="gzip")` and inspect selectively instead of printing it.

## Report

`python -m mortalmath report --focus <entry> --findings results/findings.txt` writes `results/report.html`.
Findings are hand-written, one per line, and must match the numbers. Rewrite them when the focus model or the leaderboard changes.

## Tests

`python -m pytest -q tests` runs offline in a few seconds.
