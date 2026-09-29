"""Run an external agent harness (OpenCode or Prime Agent) on AppWorld tasks with our solver model,
then grade with AppWorld's own evaluator. Same model, same tasks, same grader as run_eval.py;
only the harness differs.

    python scripts/external_eval.py --harness opencode --experiment 9b_opencode_pilot \
        --dataset dev --task-ids-file results/dev20_task_ids.txt --limit 3

Per task: start an AppWorld API server, load the task into it (AppWorld(remote_apis_url=...)),
expose the task's apps as tools via AppWorld's MCP server (OpenCode), run the harness headless,
then world.save() so the final DB state can be graded.

Task prompt: AppWorld's official function-calling agent prompt (experiments/prompts/
function_calling_agent/instructions.txt) without its worked examples, so no harness gets
instructions written by us.

Safety: the harnesses run on this machine, so their shell / file-editing / web tools are disabled;
the model can only act through AppWorld's tools.
"""

import argparse
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402  sets APPWORLD_ROOT; reuses endpoints and env loading

PROJECT_ROOT = run_eval.PROJECT_ROOT
APPWORLD_ROOT = run_eval.APPWORLD_ROOT
APPWORLD_BIN = str(PROJECT_ROOT / ".venv/bin/appworld")
PROMPT_TEMPLATE = APPWORLD_ROOT / "experiments/prompts/function_calling_agent/instructions.txt"
TASK_TIMEOUT = 20 * 60
MAX_STEPS = 50


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def render_prompt(task) -> str:
    """Official AppWorld function-calling prompt, minus the tutorial examples."""
    from jinja2 import Template

    text = PROMPT_TEMPLATE.read_text()
    header = text.split("Next, I will show you some worked-out examples")[0]
    header = Template(header).render(main_user=task.supervisor)
    app_descriptions = json.dumps(
        [{"name": k, "description": v} for k, v in task.app_descriptions.items()], indent=1)
    header = header.replace("{app_descriptions}", app_descriptions).replace("{max_steps}", str(MAX_STEPS))
    lines = [line for line in header.splitlines() if not set(line.strip()) <= set("-=") or not line.strip()]
    return "\n".join(lines).strip() + f"\n\n# Real Task Instruction\n{task.instruction}\n"


def mcp_command(apis_url: str, app_names: list[str]) -> list[str]:
    """AppWorld's MCP server behind the allow-list proxy. content_only: results as plain text,
    which every MCP client reads (the default puts them only in structuredContent)."""
    return [sys.executable, str(PROJECT_ROOT / "scripts/mcp_filter_proxy.py"),
            APPWORLD_BIN, "serve", "mcp", "stdio", "--remote-apis-url", apis_url,
            "--app-names", ",".join(app_names), "--root", str(APPWORLD_ROOT), "--output-type", "content_only"]


def predict_apis(task, args) -> tuple[list[str], dict]:
    """AppWorld's official API predictor (as used by its tool-calling agents): the solver model
    picks <= 20 APIs for the task, with AppWorld's prompt and 3 fixed train-split demos."""
    from appworld_agents.code.simplified.api_predictor import APIPredictor

    model_config = {
        "client_name": "openai", "api_type": "chat_completions", "base_url": args.base_url,
        "api_key_env_name": "SGLANG_API_KEY", "name": args.model, "temperature": 0.0,
        "max_tokens": 1500, "seed": 100, "retry_after_n_seconds": 15, "use_cache": False,
        "max_retries": 100,
        "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "input_cache_write": 0.0, "output": 0.0},
    }
    predictor = APIPredictor(
        model_config=model_config,
        prompt_file_path=str(APPWORLD_ROOT / "experiments/prompts/api_predictor.txt"),
        demo_task_ids=["82e2fac_1", "29caf6f_1", "d0b1f43_1"],
        max_predicted_apis=20, app_api_separator="__", mode="predicted")
    apis, output = predictor.predict(task)
    tokens = output["standardized_usage"].tokens
    return apis, {"input_tokens": tokens.input_cache_miss + tokens.input_cache_hit, "output_tokens": tokens.output}


def opencode_config(apis_url: str, app_names: list[str], allowed_tools: list[str], model: str,
                    base_url: str, context: int, ondemand: bool) -> dict:
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "model": f"autoharness/{model}",
        "provider": {
            "autoharness": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "AutoHarness solver",
                "options": {"baseURL": base_url, "apiKey": "{env:SGLANG_API_KEY}"},
                "models": {model: {"name": model, "limit": {"context": context, "output": 1500}}},
            }
        },
        "mcp": {
            "appworld": {
                "type": "local",
                "command": mcp_command(apis_url, app_names),
                "environment": {"ALLOWED_TOOLS": ",".join(allowed_tools), "ONDEMAND_APIS": "1" if ondemand else "0"},
                "enabled": True,
                "timeout": 60000,
            }
        },
        # Safety: no shell, file edits, or web access on this machine. The read-only filesystem tools
        # are off too: they read the *host's* files, which the model mistook for AppWorld's file_system
        # app (21 tasks in the first run, 0 solved).
        "tools": {"bash": False, "edit": False, "write": False, "patch": False, "webfetch": False,
                  "read": False, "glob": False, "grep": False, "list": False},
        "permission": {"bash": "deny", "edit": "deny", "webfetch": "deny"},
        # Same sampling as the ReAct / AutoHarness runs.
        "agent": {"build": {"temperature": 0}},
    }


def run_opencode(prompt: str, apis_url: str, app_names: list[str], allowed_tools: list[str], args,
                 log_path: Path) -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="ah_opencode_"))
    config_path = work_dir / "opencode.json"
    config_path.write_text(json.dumps(opencode_config(apis_url, app_names, allowed_tools, args.model,
                                                      args.base_url, args.context,
                                                      args.api_access == "ondemand"), indent=2))
    # Own data dir per task: parallel runs otherwise race on OpenCode's local SQLite DB.
    env = dict(os.environ, OPENCODE_CONFIG=str(config_path), XDG_DATA_HOME=str(work_dir / "data"))
    cmd = ["opencode", "run", "--format", "json", "--auto", "--model", f"autoharness/{args.model}",
           "--dir", str(work_dir), prompt]
    with log_path.open("w") as log:
        # Own process group, so the timeout also stops OpenCode's MCP children.
        # stdin=DEVNULL: `opencode run` reads piped stdin as extra message text and would wait forever
        # on an open stdin inherited from the launching shell.
        proc = subprocess.Popen(cmd, cwd=work_dir, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + TASK_TIMEOUT  # wall clock
        while proc.poll() is None:
            if time.time() > deadline:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                log.write('\n{"type": "autoharness_timeout"}\n')
                break
            time.sleep(2)
    shutil.rmtree(work_dir, ignore_errors=True)


def solve(task_id: str, args) -> dict | None:
    from appworld import AppWorld

    if args.start_deadline and time.time() > args.start_deadline:
        return None  # budget guard: don't start new tasks after the deadline

    port = free_port()
    server = subprocess.Popen([APPWORLD_BIN, "serve", "apis", "--port", str(port), "--root", str(APPWORLD_ROOT),
                               "--no-show-usage"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    apis_url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):  # wait for the API server
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        start = time.time()
        with AppWorld(task_id=task_id, experiment_name=args.experiment, remote_apis_url=apis_url) as world:
            prompt = render_prompt(world.task)
            allowed_tools, predictor_usage = predict_apis(world.task, args)
            app_names = sorted({tool.split("__")[0] for tool in allowed_tools})
            if args.api_access == "ondemand":
                # Every app the task allows is reachable; predicted APIs stay as direct tools.
                app_names = [a for a in world.task.allowed_apps if a not in ("admin", "api_docs")]
                prompt += ("\nIf an API you need is not in your tool list, look it up with "
                           "api_docs__show_api_descriptions / api_docs__show_api_doc and call it with call_api.\n")
            log_path = Path(world.output_logs_directory) / f"{args.harness}_events.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            (log_path.parent / "prompt.txt").write_text(prompt)
            (log_path.parent / "predicted_apis.txt").write_text("\n".join(allowed_tools) + "\n")
            if args.harness == "opencode":
                run_opencode(prompt, apis_url, app_names, allowed_tools, args, log_path)
            else:
                raise NotImplementedError(args.harness)
            world.save()
        seconds = round(time.time() - start, 1)
    finally:
        server.terminate()
    print(f"  done {task_id} in {seconds}s ({len(allowed_tools)} tools)", flush=True)
    return {"task_id": task_id, "seconds": seconds, "log": str(log_path), "predictor": predictor_usage,
            "num_tools": len(allowed_tools)}


def opencode_stats(log_path: str) -> dict:
    """Sum token usage and count steps/tool calls from OpenCode's JSON event stream."""
    stats = {"input_tokens": 0, "output_tokens": 0, "steps": 0, "tool_calls": 0, "appworld_tool_calls": 0, "timeout": False}
    for line in Path(log_path).read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        part = event.get("part", {}) or {}
        if kind == "step_finish":
            tokens = part.get("tokens", {}) or {}
            cache = tokens.get("cache", {}) or {}
            stats["input_tokens"] += tokens.get("input", 0) + cache.get("read", 0) + cache.get("write", 0)
            stats["output_tokens"] += tokens.get("output", 0) + tokens.get("reasoning", 0)
            stats["steps"] += 1
        elif kind == "tool_use":
            stats["tool_calls"] += 1
            # AppWorld tools come from the "appworld" MCP server; anything else is a built-in or invalid.
            if str(part.get("tool", "")).startswith("appworld_") and part.get("state", {}).get("status") != "error":
                stats["appworld_tool_calls"] += 1
        elif kind == "autoharness_timeout":
            stats["timeout"] = True
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", choices=["opencode", "prime"], required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset", default="dev", choices=["train", "dev", "test_normal"])
    parser.add_argument("--task-ids-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="only the first N tasks (pilot)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default="qwen3.5-9b-64k", choices=sorted(run_eval.ENDPOINTS),
                        help="external harnesses use the 64k-context server by default")
    parser.add_argument("--context", type=int, help="context window told to the harness (default: server's)")
    parser.add_argument("--api-access", choices=["predicted", "ondemand"], default="ondemand",
                        help="predicted: only AppWorld's predicted APIs (first run); "
                             "ondemand: predicted APIs + doc lookup + call_api for any API")
    parser.add_argument("--shuffle-seed", type=int, help="run tasks in a shuffled order")
    parser.add_argument("--start-deadline-min", type=float,
                        help="budget guard: start no new task after this many minutes")
    args = parser.parse_args()
    args.base_url = run_eval.ENDPOINTS[args.model]
    args.context = args.context or (65536 if args.model.endswith("-64k") else 32768)

    run_eval.load_env_file()
    from appworld.evaluator import evaluate_tasks

    task_ids = args.task_ids_file.read_text().split()[: args.limit]
    if args.shuffle_seed is not None:
        # Shuffled order, so tasks skipped by the budget guard leave a random (fair) subset.
        random.Random(args.shuffle_seed).shuffle(task_ids)
    out_dir = run_eval.RESULTS_DIR / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_ids.txt").write_text("\n".join(task_ids) + "\n")
    print(f"Running {len(task_ids)} {args.dataset} tasks with {args.harness}, {args.workers} workers", flush=True)

    run_eval.wait_for_server(args.base_url)
    wall_start = time.time()
    args.start_deadline = wall_start + 60 * args.start_deadline_min if args.start_deadline_min else None
    # First task alone warms shared caches (e.g. OpenCode's provider package), then the rest in parallel.
    # Separate processes: AppWorld keeps per-process global state (e.g. the frozen task clock).
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = [pool.submit(solve, task_ids[0], args).result()]
        runs += list(pool.map(partial(solve, args=args), task_ids[1:]))
    wall_seconds = round(time.time() - wall_start, 1)
    skipped = [t for t, r in zip(task_ids, runs) if r is None]
    runs = [r for r in runs if r is not None]
    if skipped:
        print(f"Budget guard: {len(skipped)} tasks not started; results cover {len(runs)} tasks.", flush=True)
    task_ids = [r["task_id"] for r in runs]

    individual = evaluate_tasks(task_ids, experiment_name=args.experiment, save_reports=True)["individual"]
    rows = []
    for r in runs:
        result = individual.get(r["task_id"], {})
        passes, failures = len(result.get("passes", [])), len(result.get("failures", []))
        rows.append({"task_id": r["task_id"], "passed": bool(result.get("success", False)),
                     "tests_passed": passes, "tests_total": passes + failures,
                     "seconds": r["seconds"], "num_tools": r["num_tools"], **opencode_stats(r["log"])})
        # The API-prediction call is part of this harness's cost.
        rows[-1]["input_tokens"] += r["predictor"]["input_tokens"]
        rows[-1]["output_tokens"] += r["predictor"]["output_tokens"]
    n = len(rows)
    summary = {
        "experiment": args.experiment, "dataset": args.dataset, "harness": args.harness,
        "model": args.model, "context": args.context, "api_access": args.api_access, "num_tasks": n,
        "skipped_by_budget_guard": skipped,
        "pass_at_1": round(100 * sum(r["passed"] for r in rows) / n, 1),
        "test_pass_rate": round(100 * sum(r["tests_passed"] for r in rows)
                                / max(1, sum(r["tests_total"] for r in rows)), 1),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / n),
        "avg_output_tokens": round(sum(r["output_tokens"] for r in rows) / n),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 1),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
        "timeouts": sum(r["timeout"] for r in rows),
        "wall_seconds": wall_seconds,
        "tasks": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for r in rows:
        print(f"{r['task_id']:<14}{'✓' if r['passed'] else '✗'} tests {r['tests_passed']}/{r['tests_total']} "
              f"in_tok {r['input_tokens']} steps {r['steps']} tools {r['tool_calls']} appworld_tools {r['appworld_tool_calls']} {r['seconds']}s")
    print(f"\npass@1 {summary['pass_at_1']}%  |  tests passed {summary['test_pass_rate']}%  |  "
          f"avg input tokens {summary['avg_input_tokens']}  |  avg steps {summary['avg_steps']}  |  "
          f"timeouts {summary['timeouts']}  |  wall {wall_seconds}s")


if __name__ == "__main__":
    main()
