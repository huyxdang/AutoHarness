"""Run AppWorld tasks on Modal CPU containers instead of local processes.

Each task runs in its own container (same code path as the local runner: scripts/run_eval.py's
solve_chunk), and its output folder is shipped back into appworld/experiments/outputs/ so grading
and the optimizer work unchanged. harness/ and scripts/ are mounted at launch, so every run uses
the current harness.

Used by scripts/run_eval.py --backend modal. Smoke test:
    MODAL_PROFILE=dangxhwee2003 python scripts/modal_eval.py --experiment modal_smoke --n 2
"""

import io
import os
import tarfile
import time
from pathlib import Path

# Always bill this project's Modal workspace, whatever profile is globally active.
os.environ.setdefault("MODAL_PROFILE", "dangxhwee2003")
import modal  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_APPWORLD = PROJECT_ROOT / "appworld"
REMOTE_PROJECT = "/root/project"
REMOTE_APPWORLD = f"{REMOTE_PROJECT}/appworld"
MAX_CONTAINERS = 48

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_dir(LOCAL_APPWORLD, REMOTE_APPWORLD, copy=True,
                   ignore=[".git", "experiments/outputs", "generate", "notebooks", "images", "tests",
                           "**/__pycache__", "**/.cache"])
    .run_commands(f"pip install -q -e {REMOTE_APPWORLD} -e '{REMOTE_APPWORLD}/experiments[simplified]'")
    .env({"APPWORLD_ROOT": REMOTE_APPWORLD, "PYTHONPATH": REMOTE_PROJECT})
    # Mounted at launch (not baked): the harness changes every optimizer round.
    .add_local_dir(PROJECT_ROOT / "harness", f"{REMOTE_PROJECT}/harness",
                   ignore=["**/__pycache__"])
    .add_local_dir(PROJECT_ROOT / "scripts", f"{REMOTE_PROJECT}/scripts",
                   ignore=["**/__pycache__"])
)

app = modal.App("autoharness-eval")


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    max_containers=MAX_CONTAINERS,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("autoharness-sglang")],
)
def solve_task(agent_config: dict, experiment: str, task_id: str, base_url: str,
               harness_files: dict[str, str] | None = None) -> tuple[str, float, bytes]:
    import shutil
    import sys

    os.environ["MODEL_SERVER_URL"] = base_url
    if harness_files:
        # Candidate harness sent with the call, so several candidates can be scored at once.
        # Written to its own directory (mounts are read-only) that shadows the mounted package;
        # must happen before run_eval imports `harness`.
        override_root = Path("/root/candidate")
        harness_dir = override_root / "harness"
        harness_dir.mkdir(parents=True, exist_ok=True)
        for name, content in harness_files.items():
            (harness_dir / name).write_text(content)
        sys.path.insert(0, str(override_root))
        agent_config = dict(agent_config, prompt_file_path=str(harness_dir / "prompt.txt"))
    sys.path.insert(0, f"{REMOTE_PROJECT}/scripts")
    import run_eval

    import harness

    print(f"{task_id}: harness from {Path(harness.__file__).parent}", flush=True)
    task_dir = Path(REMOTE_APPWORLD) / "experiments/outputs" / experiment / "tasks" / task_id
    shutil.rmtree(task_dir, ignore_errors=True)  # containers are reused across tasks
    seconds = run_eval.solve_chunk(agent_config, experiment, [task_id])[task_id]

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(task_dir, arcname=task_id)
    return task_id, seconds, buffer.getvalue()


def read_harness_files(harness_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in harness_dir.iterdir() if p.suffix in (".py", ".txt")}


def run_tasks_remote(agent_config: dict, experiment: str, task_ids: list[str], base_url: str,
                     harness_dir: Path | None = None) -> dict[str, float]:
    """Run tasks on Modal and unpack their outputs locally. Returns seconds per task.
    harness_dir: score a candidate harness (a copy of harness/) instead of the committed one."""
    local_tasks_dir = LOCAL_APPWORLD / "experiments/outputs" / experiment / "tasks"
    local_tasks_dir.mkdir(parents=True, exist_ok=True)
    harness_files = read_harness_files(harness_dir) if harness_dir else None
    seconds: dict[str, float] = {}
    with modal.enable_output(), app.run():
        calls = [(agent_config, experiment, task_id, base_url, harness_files) for task_id in task_ids]
        for task_id, task_seconds, archive in solve_task.starmap(calls, order_outputs=False):
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                tar.extractall(local_tasks_dir, filter="data")
            seconds[task_id] = task_seconds
            print(f"  done {task_id} in {task_seconds}s ({len(seconds)}/{len(task_ids)})", flush=True)
    return seconds


if __name__ == "__main__":
    import argparse
    import random
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import run_eval

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="modal_smoke")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--model", default="qwen3.5-9b")
    args = parser.parse_args()

    run_eval.load_env_file()
    from appworld.task import load_task_ids

    task_ids = sorted(random.Random(1).sample(load_task_ids("dev"), args.n))
    config_args = argparse.Namespace(agent_type="simplified_react_code_agent", model=args.model,
                                     base_url=run_eval.ENDPOINTS[args.model], max_tokens=1500, max_steps=50,
                                     prompt_file=run_eval.DEFAULT_PROMPT)
    agent_config = run_eval.build_agent_config(config_args)
    # The prompt path must resolve inside the container.
    agent_config["prompt_file_path"] = agent_config["prompt_file_path"].replace(str(PROJECT_ROOT), REMOTE_PROJECT)
    start = time.time()
    print(run_tasks_remote(agent_config, args.experiment, task_ids, run_eval.ENDPOINTS[args.model]))
    print(f"wall {time.time() - start:.0f}s; outputs in {LOCAL_APPWORLD / 'experiments/outputs' / args.experiment}")
