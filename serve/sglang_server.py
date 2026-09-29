"""SGLang server for the solver model on Modal (OpenAI-compatible API).

Adapted from modal-labs/modal-examples 06_gpu_and_ml/llm-serving/sglang_low_latency.py,
trimmed down for a single small GPU and a tight budget.

Deploy:  SOLVER=qwen3.5-4b MODAL_PROFILE=dangxhwee2003 modal deploy serve/sglang_server.py
         SOLVER=qwen3.5-9b MODAL_PROFILE=dangxhwee2003 modal deploy serve/sglang_server.py
Each solver is its own Modal app, so switching models never breaks the other endpoint.
"""

import os
import subprocess
import time

import modal

MINUTES = 60

SOLVERS = {
    "qwen3.5-4b": {  # app "autoharness-sglang" (original name kept so the endpoint URL is stable)
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "gpu": "A10",  # 24 GB, ~$1.10/h
        "app": "autoharness-sglang",
    },
    "qwen3.5-9b": {  # ~19 GB bf16 weights: needs the 48 GB L40S for KV-cache headroom
        "model": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "gpu": "L40S",  # 48 GB, ~$1.95/h
        "app": "autoharness-sglang-9b",
    },
    # Same model with a 64k context, for external harnesses (OpenCode, Prime Agent) whose own system
    # prompts and tool schemas need more room. Separate app so the 32k setup behind our results is untouched.
    "qwen3.5-9b-64k": {
        "model": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "gpu": "L40S",
        "app": "autoharness-sglang-9b-64k",
        "context": 65536,
    },
}
SOLVER = os.environ.get("SOLVER", "qwen3.5-4b")
MODEL_NAME = SOLVERS[SOLVER]["model"]
MODEL_REVISION = SOLVERS[SOLVER]["revision"]
SERVED_MODEL_NAME = SOLVER
GPU = SOLVERS[SOLVER]["gpu"]
# 32k by default: AppWorld prompts are capped at ~50k chars, and a small window keeps the KV cache roomy.
CONTEXT_LENGTH = SOLVERS[SOLVER].get("context", 32768)
TARGET_INPUTS = 32  # concurrent AppWorld tasks per container (L40S KV cache was only ~35% used at 16)
PORT = 8000

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.12.post1-cu130")
    .entrypoint([])
    .run_commands("rm -rf /root/.cache/huggingface")
)

HF_CACHE_VOL = modal.Volume.from_name("autoharness-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
sglang_image = sglang_image.env(
    # SOLVER is baked into the image so the container resolves the same model as the deploy.
    {"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1", "SOLVER": SOLVER}
)

# SGLANG_API_KEY: requests must send `Authorization: Bearer <key>`, so a leaked URL can't burn credits.
API_KEY_SECRET = modal.Secret.from_name("autoharness-sglang")

app = modal.App(name=SOLVERS[SOLVER]["app"])

with sglang_image.imports():
    import requests


def write_no_thinking_chat_template() -> str:
    """Qwen3.5 thinks by default. AppWorld's client can't pass chat_template_kwargs,
    so force enable_thinking=false in the template itself."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    path = "/tmp/chat_template_no_thinking.jinja"
    with open(path, "w") as f:
        f.write("{%- set enable_thinking = false %}\n" + tokenizer.chat_template)
    return path


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES) -> None:
    deadline = time.time() + timeout
    headers = {"Authorization": f"Bearer {os.environ['SGLANG_API_KEY']}"}
    while time.time() < deadline:
        if (rc := process.poll()) is not None:
            raise subprocess.CalledProcessError(rc, cmd=process.args)
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health", headers=headers).raise_for_status()
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
            time.sleep(5)
    raise TimeoutError(f"SGLang server not ready within {timeout} seconds")


@app.server(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[API_KEY_SECRET],
    min_containers=0,
    max_containers=1,  # never pay for a second GPU
    scaledown_window=5 * MINUTES,  # shut down after 5 idle minutes
    startup_timeout=20 * MINUTES,
    port=PORT,
    exit_grace_period=15,
    target_concurrency=TARGET_INPUTS,
    unauthenticated=True,  # protected by SGLANG_API_KEY instead of Modal proxy auth
)
class SGLang:
    @modal.enter()
    def startup(self):
        chat_template = write_no_thinking_chat_template()
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--served-model-name", SERVED_MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--api-key", os.environ["SGLANG_API_KEY"],
            "--chat-template", chat_template,
            "--context-length", str(CONTEXT_LENGTH),
            "--mem-fraction-static", "0.85",
            "--cuda-graph-max-bs", str(TARGET_INPUTS * 2),
            # Parse Qwen's XML tool calls into OpenAI tool_calls. Only used when a request passes
            # `tools` (OpenCode / Prime Agent); text-only harnesses (ReAct, ours) are unaffected.
            "--tool-call-parser", "qwen3_coder",
        ]
        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        HF_CACHE_VOL.commit()  # persist downloaded weights for faster cold starts

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def test(prompt: str = "In one sentence, what is a harness for an LLM agent?"):
    import json
    import urllib.request

    url = SGLang.get_url()
    api_key = os.environ["SGLANG_API_KEY"]
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    start = time.time()
    with urllib.request.urlopen(request, timeout=20 * MINUTES) as response:
        body = json.load(response)
    print(f"URL: {url}")
    print(f"Latency: {time.time() - start:.1f}s")
    print(f"Usage: {body['usage']}")
    print(f"Reply: {body['choices'][0]['message']['content']}")
