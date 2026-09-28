"""One round with parallel candidates: run train tasks, diagnose the top failure patterns once,
then let one Claude Code editor per pattern write a candidate harness, all in parallel.

    python optimizer/candidates.py --round 4 --tag 9b_ --n 15 --k 3

Output: optimizer/workspace/<tag>round<k>/candidates.json listing each candidate's harness copy
and rationale. Scoring and keep/reject happen in optimizer/loop_parallel.py.

Guardrails (same as round.py):
- The optimizer only sees TRAIN trajectories (plus history.md).
- Diagnosis is read-only. Each editor may edit only its own candidate copy of harness/.
- Any change to the repo outside optimizer/workspace/ is reverted.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from round import (PROJECT_ROOT, build_failure_file, run,  # noqa: E402
                   snapshot_files)

WORKSPACE = PROJECT_ROOT / "optimizer/workspace"
HISTORY = PROJECT_ROOT / "optimizer/history.md"

DIAGNOSIS_PROMPT = """\
You are diagnosing failures of a small LLM agent ({model}) on the AppWorld benchmark. The agent
solves tasks by writing Python code that calls app APIs (ReAct style). Its harness is
harness/prompt.txt (instructions + few-shot demo) and harness/react_agent.py (agent loop).

Evidence in {workspace}: failures/<task_id>.md (instruction, full trajectory, evaluation report
for each FAILED train task) and summary.json. {history} lists earlier harness edits and whether
held-out validation KEPT or REJECTED them.

Identify the {k} most common DISTINCT failure patterns that a harness change (prompt or agent loop)
could plausibly fix. Rank by number of failed tasks affected. Skip ideas the history shows were
rejected unless you have a substantially different fix. Do not edit any files.

Your final message must be ONLY a JSON array of {k} objects:
[{{"pattern": "...", "tasks": ["<task_id>", ...], "fix_idea": "..."}}]
"""

EDITOR_PROMPT = """\
You are improving the harness of a small LLM agent ({model}) on the AppWorld benchmark. The agent
writes Python code that calls app APIs (ReAct style).

Harness files you may edit (ONLY these, in your current directory):
- harness/prompt.txt       instructions + few-shot demo given to the model
- harness/react_agent.py   agent loop: prompt construction, output parsing, history handling

Evidence (read-only): {workspace}/failures/<task_id>.md for failed train tasks.

Target this failure pattern: {pattern}
Seen in: {tasks}
Starting idea (improve on it if you see a better fix): {fix_idea}

Make ONE focused change that fixes this pattern. Keep it general: never hardcode task-specific
answers, task IDs, or user data. Keep the output format compatible with the parser in
react_agent.py. Finish with exactly one line starting with `RATIONALE:` describing the pattern
(with how many failed tasks showed it) and the change, in one sentence.
"""


def claude(prompt: str, cwd: Path, tools: list[str], model: str, extra_dirs: list[Path]) -> subprocess.Popen:
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json",
           "--allowedTools", *tools, "--disallowedTools", "Bash", "WebFetch", "WebSearch"]
    for d in extra_dirs:
        cmd += ["--add-dir", str(d)]
    return subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def first_json_list(text: str) -> list[dict]:
    """First JSON array of objects in the reply (prose may contain other brackets)."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            return value
    return []


def result_text(proc: subprocess.Popen) -> str:
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed: {stderr[-2000:]}")
    return json.loads(stdout).get("result", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--tag", default="9b_")
    parser.add_argument("--n", type=int, default=15, help="train tasks to run")
    parser.add_argument("--k", type=int, default=3, help="candidates per round")
    parser.add_argument("--model", default="qwen3.5-9b", help="solver model")
    parser.add_argument("--optimizer-model", default="claude-opus-4-8[1m]")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    experiment = f"{args.tag}round{args.round}_train"
    workspace = WORKSPACE / f"{args.tag}round{args.round}"

    # 1. Current harness on fresh train tasks.
    if not args.skip_run:
        run([sys.executable, "scripts/run_eval.py", "--experiment", experiment, "--dataset", "train",
             "--n", str(args.n), "--seed", str(args.round), "--model", args.model,
             "--agent-type", "autoharness_react_code_agent",
             "--prompt-file", str(PROJECT_ROOT / "harness/prompt.txt")])

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import run_eval  # noqa: F401  sets APPWORLD_ROOT before appworld is imported

    summary = json.loads((PROJECT_ROOT / "results" / experiment / "summary.json").read_text())
    shutil.rmtree(workspace, ignore_errors=True)
    (workspace / "failures").mkdir(parents=True)
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2))
    failed = [t["task_id"] for t in summary["tasks"] if not t["passed"]]
    for task_id in failed:
        (workspace / "failures" / f"{task_id}.md").write_text(build_failure_file(experiment, task_id))
    print(f"Train pass@1 {summary['pass_at_1']}%, {len(failed)} failures -> {workspace}", flush=True)
    if not failed:
        (workspace / "candidates.json").write_text("[]")
        return

    before = snapshot_files()

    # 2. One read-only diagnosis call ranks the top-k patterns.
    history = HISTORY.relative_to(PROJECT_ROOT) if HISTORY.exists() else "(no history yet)"
    diagnosis = result_text(claude(
        DIAGNOSIS_PROMPT.format(model=args.model, workspace=workspace.relative_to(PROJECT_ROOT),
                                history=history, k=args.k),
        cwd=PROJECT_ROOT, tools=["Read", "Grep", "Glob"], model=args.optimizer_model, extra_dirs=[]))
    (workspace / "diagnosis_reply.md").write_text(diagnosis)
    patterns = first_json_list(diagnosis)[: args.k]
    (workspace / "diagnosis.json").write_text(json.dumps(patterns, indent=2))
    print(f"Diagnosis: {[p['pattern'][:80] for p in patterns]}", flush=True)

    # 3. One editor per pattern, in parallel, each on its own copy of the harness.
    procs = []
    for i, p in enumerate(patterns, start=1):
        cand_dir = workspace / f"cand{i}"
        shutil.copytree(PROJECT_ROOT / "harness", cand_dir / "harness",
                        ignore=shutil.ignore_patterns("__pycache__"))
        prompt = EDITOR_PROMPT.format(model=args.model, workspace=workspace, pattern=p["pattern"],
                                      tasks=", ".join(p.get("tasks", [])), fix_idea=p.get("fix_idea", ""))
        procs.append((i, cand_dir, claude(prompt, cwd=cand_dir,
                                          tools=["Read", "Grep", "Glob", "Edit(harness/**)", "Write(harness/**)"],
                                          model=args.optimizer_model, extra_dirs=[workspace])))

    candidates = []
    for i, cand_dir, proc in procs:
        try:
            reply = result_text(proc)
        except RuntimeError as error:
            print(f"Candidate {i} failed: {error}", flush=True)
            continue
        (cand_dir / "optimizer_reply.md").write_text(reply)
        rationale = next((line.split("RATIONALE:", 1)[1].strip() for line in reply.splitlines()
                          if "RATIONALE:" in line), "no rationale given")
        changed = any((cand_dir / "harness" / f).read_bytes() != (PROJECT_ROOT / "harness" / f).read_bytes()
                      for f in ("prompt.txt", "react_agent.py"))
        print(f"Candidate {i}: {'changed' if changed else 'NO CHANGE'} - {rationale[:160]}", flush=True)
        if changed:
            candidates.append({"id": i, "harness_dir": str(cand_dir / "harness"), "rationale": rationale})

    # 4. Guardrail: nothing in the repo outside the workspace may change.
    after = snapshot_files()
    for path in set(before) | set(after):
        if before.get(path) != after.get(path) and not path.startswith("optimizer/workspace/"):
            print(f"Reverting edit outside the workspace: {path}", flush=True)
            if path in before:
                (PROJECT_ROOT / path).write_bytes(before[path])
            else:
                (PROJECT_ROOT / path).unlink(missing_ok=True)

    (workspace / "candidates.json").write_text(json.dumps(candidates, indent=2))
    print(f"{len(candidates)} candidates -> {workspace / 'candidates.json'}", flush=True)


if __name__ == "__main__":
    main()
