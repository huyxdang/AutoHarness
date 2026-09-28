"""Optimization loop with parallel candidates and same-batch re-scoring.

    python optimizer/loop_parallel.py --start-round 4 --rounds 3

Each round:
  1. optimizer/candidates.py: train run, one diagnosis call, k editor calls in parallel -> k candidates.
  2. Score the current best AND every candidate on the same dev tasks, all at once (Modal backend),
     so they share conditions and the best's score is fresh (runs vary by ~±2 dev tasks).
  3. Winner = best candidate by (pass@1, unit-test rate) among those within the token budget.
     KEEP it only if it beats the re-scored current best by >= MARGIN_TASKS dev tasks
     (picking the max of k noisy candidates overstates the winner).
  4. Log every candidate to optimizer/history.md; commit. Stop early after PATIENCE rounds
     without a keep.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
HISTORY = PROJECT_ROOT / "optimizer/history.md"
TOKEN_TOLERANCE = 1.20
MAX_ROUNDS = 3  # N: optimization rounds per run (override with --rounds)
MARGIN_TASKS = 2
PATIENCE = 2


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, text=True)


def score_in_parallel(jobs: dict[str, Path], dev_ids: Path, model: str) -> dict[str, dict]:
    """jobs: experiment name -> harness dir. Runs all dev evaluations concurrently."""
    procs = {}
    for experiment, harness_dir in jobs.items():
        cmd = [sys.executable, "scripts/run_eval.py", "--experiment", experiment, "--dataset", "dev",
               "--task-ids-file", str(dev_ids), "--model", model, "--backend", "modal",
               "--agent-type", "autoharness_react_code_agent", "--harness-dir", str(harness_dir)]
        print("$", " ".join(cmd), flush=True)
        log = open(PROJECT_ROOT / f"results_{experiment}.log", "w")
        procs[experiment] = (subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT), log)
    for experiment, (proc, log) in procs.items():
        proc.wait()
        log.close()
        if proc.returncode != 0:
            raise RuntimeError(f"dev scoring failed for {experiment}, see results_{experiment}.log")
    return {e: json.loads((RESULTS / e / "summary.json").read_text()) for e in jobs}


def passes(s: dict) -> int:
    return sum(t["passed"] for t in s["tasks"])


def fmt(s: dict) -> str:
    return (f"pass@1 {s['pass_at_1']}% ({passes(s)}/{s['num_tasks']}), tests {s['test_pass_rate']}%, "
            f"{s['avg_input_tokens']:,} input tok/task")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-round", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--tag", default="9b_")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--dev-ids", type=Path, default=RESULTS / "dev20_task_ids.txt")
    parser.add_argument("--train-n", type=int, default=15)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    with HISTORY.open("a") as f:
        f.write(f"\n## Parallel candidates (k={args.k}, current best re-scored each round, "
                f"keep only if +{MARGIN_TASKS} dev tasks)\n")

    rounds_without_keep = 0
    for r in range(args.start_round, args.start_round + args.rounds):
        run([sys.executable, "optimizer/candidates.py", "--round", str(r), "--tag", args.tag,
             "--n", str(args.train_n), "--k", str(args.k), "--model", args.model])
        workspace = PROJECT_ROOT / f"optimizer/workspace/{args.tag}round{r}"
        candidates = json.loads((workspace / "candidates.json").read_text())
        if not candidates:
            print(f"Round {r}: no candidates; stopping.", flush=True)
            break

        best_exp = f"{args.tag}round{r}_dev_best"
        jobs = {best_exp: PROJECT_ROOT / "harness"}
        jobs |= {f"{args.tag}round{r}_dev_cand{c['id']}": Path(c["harness_dir"]) for c in candidates}
        scores = score_in_parallel(jobs, args.dev_ids, args.model)
        best = scores[best_exp]
        print(f"Round {r} current best (re-scored): {fmt(best)}", flush=True)

        eligible = []
        for c in candidates:
            s = scores[f"{args.tag}round{r}_dev_cand{c['id']}"]
            c["summary"] = s
            within = s["avg_input_tokens"] <= TOKEN_TOLERANCE * best["avg_input_tokens"]
            print(f"  candidate {c['id']}: {fmt(s)}{'' if within else '  (over token budget)'}", flush=True)
            if within:
                eligible.append(c)
        winner = max(eligible, key=lambda c: (c["summary"]["pass_at_1"], c["summary"]["test_pass_rate"]),
                     default=None)
        keep = winner is not None and passes(winner["summary"]) >= passes(best) + MARGIN_TASKS

        with HISTORY.open("a") as f:
            f.write(f"\n### Round {r} — {'KEPT candidate ' + str(winner['id']) if keep else 'no candidate kept'}\n"
                    f"- Current best re-scored: {fmt(best)}\n")
            for c in candidates:
                f.write(f"- Candidate {c['id']}: {fmt(c['summary'])} — {c['rationale']}\n")

        if keep:
            for name in ("prompt.txt", "react_agent.py"):
                shutil.copy(Path(winner["harness_dir"]) / name, PROJECT_ROOT / "harness" / name)
            run(["git", "add", "harness/"])
            run(["git", "commit", "-m", f"harness round {r} (candidate {winner['id']}): {winner['rationale']}"])
            rounds_without_keep = 0
        else:
            rounds_without_keep += 1
        verdict = f"kept candidate {winner['id']}" if keep else "no keep"
        print(f"Round {r}: {verdict}", flush=True)

        run(["git", "add", str(HISTORY), str(RESULTS), str(workspace)])
        run(["git", "commit", "-m", f"round {r} {verdict}: best re-scored {fmt(best)}"])
        if rounds_without_keep >= PATIENCE:
            print(f"Stopping: {PATIENCE} rounds without a keep.", flush=True)
            break


if __name__ == "__main__":
    main()
