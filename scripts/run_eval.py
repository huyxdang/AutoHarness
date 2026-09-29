"""Run an AppWorld agent on a sample of tasks and report pass@1, tokens, steps, and time per task.

Example (MVP baseline):
    python scripts/run_eval.py --experiment mvp_react_baseline --dataset dev --n 5 --seed 0

The agent config mirrors AppWorld's official simplified ReAct config for Qwen models
(experiments/configs/simplified_react_code_agent/alibaba/*), with the model swapped for
our SGLang endpoint on Modal.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPWORLD_ROOT = PROJECT_ROOT / "appworld"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_PROMPT = APPWORLD_ROOT / "experiments/prompts/react_code_agent/instructions.txt"
# Modal workspace that hosts the solver servers (switched from hellgod67 on 2026-09-29).
MODAL_WORKSPACE = "dangxhwee2003"
ENDPOINTS = {  # one Modal app per solver, see serve/sglang_server.py
    solver: f"https://{MODAL_WORKSPACE}--{app}-sglang.us-east.modal.direct/v1"
    for solver, app in [("qwen3.5-4b", "autoharness-sglang"), ("qwen3.5-9b", "autoharness-sglang-9b"),
                        ("qwen3.5-9b-64k", "autoharness-sglang-9b-64k"), ("qwen3.5-27b", "autoharness-sglang-27b")]
}
DEFAULT_MODEL = "qwen3.5-9b"

os.environ["APPWORLD_ROOT"] = str(APPWORLD_ROOT)


def load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    # AppWorld's LanguageModel builds a default OpenAI() client at init, which needs a key.
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("SGLANG_API_KEY", "unused"))


def build_agent_config(args: argparse.Namespace) -> dict:
    return {
        "type": args.agent_type,
        "model_config": {
            "client_name": "openai",
            "api_type": "chat_completions",
            "base_url": args.base_url,
            "api_key_env_name": "SGLANG_API_KEY",
            "name": args.model,
            "temperature": 0.0,
            # Without a cap, the 4B model sometimes rambles until the context is full (~10 min
            # per call). Same guard as AppWorld's own CI ReAct config: stop at the end of the
            # code block, and bound the reply length.
            "max_tokens": args.max_tokens,
            "stop": ["```\n"],
            "seed": 100,
            "drop_reasoning_content": False,
            "cost_per_token": {
                "input_cache_hit": 0.0,
                "input_cache_miss": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            },
            "retry_after_n_seconds": 15,  # also covers Modal's 503s during a cold start
            "use_cache": False,  # no disk cache, so timings are real
            "max_retries": 100,
        },
        "appworld_config": {"random_seed": 100, "raise_on_extra_parameters": True},
        "logger_config": {"color": False, "verbose": False},
        "usage_tracker_config": {
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "max_output_tokens_per_task": 100000,
        },
        "prompt_file_path": str(args.prompt_file),
        "ignore_multiple_calls": True,
        "max_prompt_length": None,
        "max_output_length": None,
        "max_steps": args.max_steps,
        "log_lm_calls": True,
        "skip_if_finished": False,
    }


def solve_chunk(agent_config: dict, experiment_name: str, task_ids: list[str]) -> dict[str, float]:
    """Runs in a separate process: AppWorld freezes time and holds DB state per process."""
    load_env_file()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import harness  # noqa: F401  registers our own agent types, if any
    from appworld_agents.code.simplified.agent import Agent
    from appworld_agents.code.simplified.language_model import LanguageModel

    # Compatibility shim (applies to every harness equally): SGLang returns
    # "reasoning_content": null, and AppWorld's agents call .strip() on it.
    original_generate = LanguageModel.generate

    def generate(self, *args, **kwargs):
        message = original_generate(self, *args, **kwargs)
        if message.get("reasoning_content") is None:
            message["reasoning_content"] = ""
        return message

    LanguageModel.generate = generate

    agent = Agent.from_dict(agent_config)
    seconds = {}
    for task_id in task_ids:
        start = time.time()
        agent.solve_tasks(task_ids=[task_id], experiment_name=experiment_name)
        seconds[task_id] = round(time.time() - start, 1)
    return seconds


def wait_for_server(base_url: str, timeout: int = 20 * 60) -> None:
    """Block until the Modal endpoint serves requests, so a cold start never counts toward
    per-task time (Modal returns 503 while a container boots)."""
    import httpx

    headers = {"Authorization": f"Bearer {os.environ['SGLANG_API_KEY']}"}
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"{base_url}/models", headers=headers, timeout=30).status_code == 200:
                print(f"Server ready ({time.time() - start:.0f}s)")
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"{base_url} not ready after {timeout}s")


def read_task_stats(experiment_name: str, task_id: str) -> dict:
    task_dir = APPWORLD_ROOT / "experiments/outputs" / experiment_name / "tasks" / task_id
    usage_path = task_dir / "misc/usage.json"
    tokens = json.loads(usage_path.read_text()).get("tokens", {}) if usage_path.exists() else {}
    lm_calls_path = task_dir / "logs/lm_calls.jsonl"
    steps = sum(1 for _ in lm_calls_path.open()) if lm_calls_path.exists() else 0
    return {
        "input_tokens": tokens.get("input_cache_miss", 0) + tokens.get("input_cache_hit", 0),
        "output_tokens": tokens.get("output", 0),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset", default="dev", choices=["train", "dev", "test_normal"])
    parser.add_argument("--n", type=int, default=5, help="number of tasks to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-ids-file", type=Path, help="reuse an exact task list")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--agent-type", default="simplified_react_code_agent")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1500, help="cap per model reply")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(ENDPOINTS))
    parser.add_argument("--base-url", help="override the model's endpoint")
    parser.add_argument("--backend", default="modal", choices=["modal", "local"],
                        help="modal: one Modal CPU container per task; local: processes on this machine")
    parser.add_argument("--harness-dir", type=Path,
                        help="score a candidate copy of harness/ (modal backend only)")
    args = parser.parse_args()
    if args.harness_dir and args.backend != "modal":
        parser.error("--harness-dir requires --backend modal")
    args.base_url = args.base_url or ENDPOINTS[args.model]

    load_env_file()
    # AppWorld templates base_url with this variable and fails if it is unset.
    os.environ["MODEL_SERVER_URL"] = args.base_url
    from appworld.evaluator import evaluate_tasks
    from appworld.task import load_task_ids

    if args.task_ids_file:
        task_ids = args.task_ids_file.read_text().split()
    else:
        task_ids = sorted(random.Random(args.seed).sample(load_task_ids(args.dataset), args.n))

    out_dir = RESULTS_DIR / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_ids.txt").write_text("\n".join(task_ids) + "\n")

    agent_config = build_agent_config(args)
    workers = min(args.workers, len(task_ids))
    print(f"Running {len(task_ids)} {args.dataset} tasks ({args.backend}): {task_ids}")

    wait_for_server(args.base_url)
    wall_start = time.time()
    if args.backend == "modal":
        import modal_eval

        # Paths inside the project resolve to the same place in the container.
        agent_config["prompt_file_path"] = agent_config["prompt_file_path"].replace(
            str(PROJECT_ROOT), modal_eval.REMOTE_PROJECT)
        seconds = modal_eval.run_tasks_remote(agent_config, args.experiment, task_ids, args.base_url,
                                              harness_dir=args.harness_dir)
    else:
        chunks = [task_ids[i::workers] for i in range(workers)]
        seconds = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(solve_chunk, agent_config, args.experiment, c) for c in chunks]
            for future in futures:
                seconds.update(future.result())
    wall_seconds = round(time.time() - wall_start, 1)

    metrics = evaluate_tasks(task_ids, experiment_name=args.experiment, save_reports=True)
    individual = metrics.get("individual", {})

    rows = []
    for task_id in task_ids:
        result = individual.get(task_id, {})
        num_passes, num_failures = len(result.get("passes", [])), len(result.get("failures", []))
        rows.append({"task_id": task_id, "passed": bool(result.get("success", False)),
                     "tests_passed": num_passes, "tests_total": num_passes + num_failures,
                     "seconds": seconds.get(task_id), **read_task_stats(args.experiment, task_id)})

    n = len(rows)
    summary = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "agent_type": args.agent_type,
        "prompt_file": str(args.prompt_file),
        "harness_dir": str(args.harness_dir) if args.harness_dir else None,
        "backend": args.backend,
        "model": args.model,
        "num_tasks": n,
        "pass_at_1": round(100 * sum(r["passed"] for r in rows) / n, 1),
        # Secondary signal: share of AppWorld unit tests passed (partial credit, tiebreaker only).
        "test_pass_rate": round(100 * sum(r["tests_passed"] for r in rows)
                                / max(1, sum(r["tests_total"] for r in rows)), 1),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / n),
        "avg_output_tokens": round(sum(r["output_tokens"] for r in rows) / n),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 1),
        "avg_seconds": round(sum(r["seconds"] or 0 for r in rows) / n, 1),
        "wall_seconds": wall_seconds,
        "tasks": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'task_id':<14}{'pass':>6}{'input_tok':>11}{'out_tok':>9}{'steps':>7}{'sec':>8}")
    for r in rows:
        print(f"{r['task_id']:<14}{'✓' if r['passed'] else '✗':>6}{r['input_tokens']:>11}"
              f"{r['output_tokens']:>9}{r['steps']:>7}{r['seconds']:>8}")
    print(f"\npass@1 {summary['pass_at_1']}%  |  tests passed {summary['test_pass_rate']}%"
          f"  |  avg input tokens {summary['avg_input_tokens']}"
          f"  |  avg steps {summary['avg_steps']}  |  avg sec/task {summary['avg_seconds']}"
          f"  |  wall {wall_seconds}s")
    print(f"Saved {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
