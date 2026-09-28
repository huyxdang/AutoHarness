"""One optimization round: run the current harness on train tasks, hand the failed trajectories
to Claude Code (headless), let it edit harness/ only, and commit the change with its rationale.

Example (MVP):
    python optimizer/round.py --round 1 --n 10 --seed 0

Guardrails:
- The optimizer only ever sees TRAIN trajectories. Dev/test outputs are never copied into its workspace.
- `claude -p` may read the workspace and edit files in harness/ only; any change outside harness/
  is reverted before committing.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPWORLD_OUTPUTS = PROJECT_ROOT / "appworld/experiments/outputs"
WORKSPACE = PROJECT_ROOT / "optimizer/workspace"
MAX_TRAJECTORY_CHARS = 30_000  # keep each failure file readable for the optimizer


def history_path(tag: str) -> Path:
    """The optimizer's memory: one file per optimization run (keyed by its tag, e.g. 9b_ ->
    optimizer/history_9b.md), so a new run never sees another run's edits and verdicts."""
    return PROJECT_ROOT / "optimizer" / f"history_{tag.rstrip('_') or 'default'}.md"

OPTIMIZER_PROMPT = """\
You are optimizing the harness of a small LLM agent ({model}) on the AppWorld benchmark.
The agent solves tasks by writing Python code that calls app APIs (ReAct style).

Harness files you may edit (and ONLY these):
- harness/prompt.txt       the instructions + few-shot demo given to the model
- harness/react_agent.py   the agent loop: prompt construction, output parsing, history handling

Evidence: {workspace} contains, for round {round}:
- summary.json             pass/fail, tokens, steps per train task
- failures/<task_id>.md    for each FAILED task: the task instruction, the agent's full trajectory
                           (code it ran + environment output), and the evaluation report
- {history}  earlier harness edits in this run and whether held-out validation KEPT or
                           REJECTED them (if present). Do not repeat a rejected idea; build on kept ones.

Do this:
1. Read the failures. Identify the 1-2 most common, harness-fixable failure patterns
   (e.g. malformed code blocks, not reading API docs, context overflow, not calling complete_task,
   losing track of the goal, repeated failing calls). Ignore one-off reasoning mistakes.
2. Make ONE focused change to the harness that addresses the top pattern. Keep it general:
   never hardcode task-specific answers, task IDs, or user data.
3. Keep the model's output format compatible with the parser in react_agent.py.
4. Finish with a final message containing exactly one line starting with `RATIONALE:` that states
   the failure pattern (with how many failed tasks showed it) and the change you made, in one sentence.
"""


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, text=True, **kwargs)


def snapshot_files() -> dict[str, bytes]:
    """Contents of every tracked or untracked (non-ignored) file in the repo."""
    listed = run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                 capture_output=True).stdout.splitlines()
    return {p: (PROJECT_ROOT / p).read_bytes() for p in listed if (PROJECT_ROOT / p).is_file()}


def build_failure_file(experiment: str, task_id: str) -> str:
    from appworld.task import Task

    task_dir = APPWORLD_OUTPUTS / experiment / "tasks" / task_id
    trajectory = (task_dir / "logs/environment_io.md").read_text(errors="replace")
    if len(trajectory) > MAX_TRAJECTORY_CHARS:
        half = MAX_TRAJECTORY_CHARS // 2
        trajectory = trajectory[:half] + "\n\n[... trajectory truncated ...]\n\n" + trajectory[-half:]
    report_path = task_dir / "evaluation/report.md"
    report = report_path.read_text() if report_path.exists() else "(no report)"
    instruction = Task.load(task_id=task_id).instruction
    return (f"# Task {task_id}\n\n## Instruction\n{instruction}\n\n"
            f"## Trajectory\n{trajectory}\n\n## Evaluation report\n{report}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--n", type=int, default=10, help="train tasks to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--skip-run", action="store_true", help="reuse existing train outputs")
    parser.add_argument("--model", default="qwen3.5-9b", help="solver model")
    # Pinned so a change to the global Claude Code default never silently changes the optimizer.
    parser.add_argument("--optimizer-model", default="claude-opus-4-8[1m]")
    parser.add_argument("--tag", default="", help="prefix for experiment names, e.g. 9b_")
    args = parser.parse_args()

    experiment = f"{args.tag}round{args.round}_train"
    python = sys.executable

    # 1. Run the current harness on train tasks.
    if not args.skip_run:
        run([python, "scripts/run_eval.py", "--experiment", experiment, "--dataset", "train",
             "--n", str(args.n), "--seed", str(args.seed), "--workers", str(args.workers),
             "--model", args.model, "--agent-type", "autoharness_react_code_agent",
             "--prompt-file", str(PROJECT_ROOT / "harness/prompt.txt")])

    # 2. Build the optimizer's workspace from failed train tasks only.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import run_eval  # noqa: F401  sets APPWORLD_ROOT before appworld is imported

    summary = json.loads((PROJECT_ROOT / "results" / experiment / "summary.json").read_text())
    workspace = WORKSPACE / f"{args.tag}round{args.round}"
    shutil.rmtree(workspace, ignore_errors=True)
    (workspace / "failures").mkdir(parents=True)
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2))
    failed = [t["task_id"] for t in summary["tasks"] if not t["passed"]]
    for task_id in failed:
        (workspace / "failures" / f"{task_id}.md").write_text(build_failure_file(experiment, task_id))
    print(f"Train pass@1 {summary['pass_at_1']}%, {len(failed)} failures -> {workspace}")
    if not failed:
        print("No failures to learn from; nothing to optimize.")
        return

    # 3. Let Claude Code edit the harness (headless, restricted tools).
    before = snapshot_files()
    prompt = OPTIMIZER_PROMPT.format(history=history_path(args.tag).relative_to(PROJECT_ROOT),
                                     workspace=workspace.relative_to(PROJECT_ROOT),
                                     round=args.round, model=args.model)
    result = run(["claude", "-p", prompt,
                  "--model", args.optimizer_model,
                  "--output-format", "json",
                  "--allowedTools", "Read", "Grep", "Glob",
                  "Edit(harness/**)", "Write(harness/**)",
                  "--disallowedTools", "Bash", "WebFetch", "WebSearch"],
                 capture_output=True)
    reply = json.loads(result.stdout).get("result", "")
    (workspace / "optimizer_reply.md").write_text(reply)
    rationale = next((line.split("RATIONALE:", 1)[1].strip() for line in reply.splitlines()
                      if "RATIONALE:" in line), "no rationale given")

    # 4. Enforce the guardrail: only harness/ may change. Compare against the snapshot taken just
    #    before the optimizer ran, so uncommitted edits made by us are never touched.
    after = snapshot_files()
    touched = [p for p in set(before) | set(after)
               if before.get(p) != after.get(p) and not p.startswith(("harness/", "optimizer/workspace/"))]
    for path in touched:
        print(f"Reverting optimizer edit outside harness/: {path}")
        if path in before:
            (PROJECT_ROOT / path).write_bytes(before[path])
        else:
            (PROJECT_ROOT / path).unlink(missing_ok=True)

    harness_diff = run(["git", "diff", "--stat", "--", "harness/"], capture_output=True).stdout
    if not harness_diff.strip():
        print("Optimizer made no harness change.")
        return
    print(harness_diff)
    (workspace / "rationale.txt").write_text(rationale + "\n")
    run(["git", "add", "harness/", str(workspace.relative_to(PROJECT_ROOT) / "rationale.txt"),
         str(workspace.relative_to(PROJECT_ROOT) / "optimizer_reply.md")])
    run(["git", "commit", "-m", f"harness round {args.round}: {rationale}"])
    print(f"Committed. RATIONALE: {rationale}")


if __name__ == "__main__":
    main()
