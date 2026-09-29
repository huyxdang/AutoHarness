"""Full optimization loop with automatic keep/reject on a fixed dev set.

    python optimizer/loop.py --rounds 3 --model qwen3.5-9b --tag 9b_

Round 0 scores the current harness on dev (the "best so far"). Each round then:
  1. optimizer/round.py: run train tasks (fresh sample per round), Claude Code edits harness/, commit.
  2. Score the edited harness on the same dev tasks.
  3. KEEP if pass@1 rises (or ties with a higher unit-test pass rate) and avg input tokens rise
     no more than TOKEN_TOLERANCE; otherwise REJECT and git-revert the edit.
  4. Record the decision in this run's history file (optimizer/history_<tag>.md), which the
     optimizer reads next round.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from round import history_path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
TOKEN_TOLERANCE = 1.20
MAX_ROUNDS = 3  # N: optimization rounds per run (override with --rounds)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, text=True, **kwargs)


def git_head() -> str:
    return run(["git", "rev-parse", "HEAD"], capture_output=True).stdout.strip()


def eval_dev(experiment: str, args: argparse.Namespace) -> dict:
    run([sys.executable, "scripts/run_eval.py", "--experiment", experiment, "--dataset", "dev",
         "--task-ids-file", str(args.dev_ids), "--workers", str(args.workers),
         "--model", args.model, "--agent-type", "autoharness_react_code_agent",
         "--prompt-file", str(PROJECT_ROOT / args.harness_dir / "prompt.txt"),
         "--harness-dir", str(PROJECT_ROOT / args.harness_dir)])
    return json.loads((RESULTS / experiment / "summary.json").read_text())


def fmt(s: dict) -> str:
    return (f"pass@1 {s['pass_at_1']}%, tests {s['test_pass_rate']}%, "
            f"{s['avg_input_tokens']:,} input tok/task, {s['avg_steps']} steps/task")


def decide(new: dict, best: dict) -> tuple[bool, str]:
    better = (new["pass_at_1"] > best["pass_at_1"]
              or (new["pass_at_1"] == best["pass_at_1"] and new["test_pass_rate"] > best["test_pass_rate"]))
    within_budget = new["avg_input_tokens"] <= TOKEN_TOLERANCE * best["avg_input_tokens"]
    if better and within_budget:
        return True, "better on dev and within token budget"
    if not better:
        return False, "no improvement on dev"
    return False, f"improved but input tokens rose more than {int((TOKEN_TOLERANCE - 1) * 100)}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--tag", default="9b_")
    parser.add_argument("--dev-ids", type=Path, default=RESULTS / "dev20_task_ids.txt")
    parser.add_argument("--train-n", type=int, default=15)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--harness-dir", default="harness",
                        help="harness being optimized, e.g. harnesses/27b (one per model, so runs never collide)")
    args = parser.parse_args()
    HISTORY = history_path(args.tag)

    best_path = RESULTS / f"{args.tag}round0_dev" / "summary.json"
    if best_path.exists() and args.start_round > 1:
        best = json.loads(best_path.read_text())
    else:
        best = eval_dev(f"{args.tag}round0_dev", args)
        with HISTORY.open("a") as f:
            f.write(f"\n## {args.model} — round 0 (starting harness)\nDev: {fmt(best)}\n")
    print(f"Round 0 dev: {fmt(best)}", flush=True)

    for k in range(args.start_round, args.start_round + args.rounds):
        head_before = git_head()
        run([sys.executable, "optimizer/round.py", "--round", str(k), "--n", str(args.train_n),
             "--seed", str(k), "--workers", str(args.workers), "--model", args.model, "--tag", args.tag,
             "--harness-dir", args.harness_dir])
        if git_head() == head_before:
            print(f"Round {k}: no harness change; stopping.")
            break
        rationale = (PROJECT_ROOT / f"optimizer/workspace/{args.tag}round{k}/rationale.txt").read_text().strip()

        new = eval_dev(f"{args.tag}round{k}_dev", args)
        keep, reason = decide(new, best)
        verdict = "KEPT" if keep else "REJECTED"
        print(f"Round {k}: {verdict} ({reason}). Before: {fmt(best)} | After: {fmt(new)}", flush=True)
        if keep:
            best = new
        else:
            run(["git", "revert", "--no-edit", "HEAD"])

        with HISTORY.open("a") as f:
            f.write(f"\n### Round {k} — {verdict} ({reason})\n- Edit: {rationale}\n- Dev after edit: {fmt(new)}\n"
                    f"- Best dev so far: {fmt(best)}\n")
        # Only this run's results, so concurrent experiments never end up in this commit.
        run(["git", "add", str(HISTORY), *[str(p) for p in sorted(RESULTS.glob(f"{args.tag}*"))]])
        run(["git", "commit", "-m", f"round {k} {verdict.lower()}: dev {fmt(new)}"])

    print(f"\nFinal best dev: {fmt(best)}")


if __name__ == "__main__":
    main()
