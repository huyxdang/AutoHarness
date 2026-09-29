"""Package what a customer receives: the optimized harness, its run config, a results report,
and the change log, in one folder.

    python scripts/export.py --name appworld-qwen3.5-9b \
        --baseline results/9b_test_react --optimized results/9b_test_auto
    python scripts/export.py --name appworld-qwen3.5-27b --history optimizer/history_27b.md \
        --baseline results/27b_test_react --optimized results/27b_test_auto
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = """\
# Run config for the exported harness. The harness was optimized for exactly this setup;
# changing the model or sampling settings means re-validating it.
solver:
  model: {hf_model}            # Hugging Face id
  served_name: {model}
  serving: SGLang, chat template forced to enable_thinking=false
sampling:
  temperature: 0.0
  max_tokens: 1500              # per model reply
  stop: ["```\\n"]              # end of the code block
agent:
  type: autoharness_react_code_agent   # harness/react_agent.py
  prompt: harness/prompt.txt
  max_steps: 50
"""

HF_MODELS = {"qwen3.5-9b": "Qwen/Qwen3.5-9B", "qwen3.5-4b": "Qwen/Qwen3.5-4B", "qwen3.5-27b": "Qwen/Qwen3.5-27B"}


def row(label: str, s: dict) -> str:
    passes = sum(t["passed"] for t in s["tasks"])
    return (f"| {label} | **{s['pass_at_1']}%** ({passes}/{s['num_tasks']}) | {s['test_pass_rate']}% | "
            f"{s['avg_input_tokens']:,} | {s['avg_output_tokens']:,} | {s['avg_steps']} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="results dir of the baseline run")
    parser.add_argument("--optimized", type=Path, required=True, help="results dir of the optimized run")
    parser.add_argument("--history", type=Path, default=Path("optimizer/history_9b.md"),
                        help="the optimization run's history file (becomes CHANGELOG.md)")
    args = parser.parse_args()

    base = json.loads((PROJECT_ROOT / args.baseline / "summary.json").read_text())
    opt = json.loads((PROJECT_ROOT / args.optimized / "summary.json").read_text())
    out = PROJECT_ROOT / "export" / args.name
    shutil.rmtree(out, ignore_errors=True)
    (out / "harness").mkdir(parents=True)
    harness_dir = PROJECT_ROOT / (opt.get("harness_dir") or "harness")  # the harness the optimized run used
    for name in ("prompt.txt", "react_agent.py"):
        shutil.copy(harness_dir / name, out / "harness" / name)

    model = opt["model"]
    (out / "config.yaml").write_text(CONFIG.format(hf_model=HF_MODELS.get(model, model), model=model))

    b = {t["task_id"]: t["passed"] for t in base["tasks"]}
    o = {t["task_id"]: t["passed"] for t in opt["tasks"]}
    only_opt = sum(o[k] and not b[k] for k in o)
    only_base = sum(b[k] and not o[k] for k in o)
    report = f"""# Results: {args.name}

Held-out evaluation on AppWorld `{opt['dataset']}` ({opt['num_tasks']} tasks, one attempt each).
Same model and sampling settings for both rows; only the harness differs.

| Harness | pass@1 | Unit tests passed | Input tokens / task | Output tokens / task | Steps / task |
|---|---|---|---|---|---|
{row('Baseline (AppWorld ReAct)', base)}
{row('Optimized (this harness)', opt)}

Task by task: {only_opt} tasks solved only with the optimized harness, {only_base} only with the baseline.

See CHANGELOG.md for every harness edit that was tried and why it was kept or rejected.
"""
    (out / "report.md").write_text(report)
    shutil.copy(PROJECT_ROOT / args.history, out / "CHANGELOG.md")
    print(f"Exported to {out}")


if __name__ == "__main__":
    main()
