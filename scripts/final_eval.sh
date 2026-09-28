#!/usr/bin/env bash
# Noise check on dev, then the headline comparison on test_normal (run once per harness).
# Baseline = AppWorld's own simplified ReAct agent + prompt, untouched.
# Auto-harness = harness/ as kept by the optimization loop.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=qwen3.5-9b
BASELINE=(--agent-type simplified_react_code_agent)
AUTO=(--agent-type autoharness_react_code_agent --prompt-file harness/prompt.txt)

# 1. Noise check: repeat both harnesses on the same 20 dev tasks.
$PY scripts/run_eval.py --experiment 9b_noise_react_dev --dataset dev \
    --task-ids-file results/dev20_task_ids.txt --workers 10 --model $MODEL "${BASELINE[@]}"
$PY scripts/run_eval.py --experiment 9b_noise_auto_dev --dataset dev \
    --task-ids-file results/dev20_task_ids.txt --workers 10 --model $MODEL "${AUTO[@]}"

# 2. Headline: all 168 test_normal tasks, once per harness.
$PY -c "import os; os.environ['APPWORLD_ROOT']='appworld'; from appworld.task import load_task_ids; \
open('results/test_normal_task_ids.txt','w').write('\n'.join(load_task_ids('test_normal'))+'\n')"
$PY scripts/run_eval.py --experiment 9b_test_react --dataset test_normal \
    --task-ids-file results/test_normal_task_ids.txt --workers 16 --model $MODEL "${BASELINE[@]}"
$PY scripts/run_eval.py --experiment 9b_test_auto --dataset test_normal \
    --task-ids-file results/test_normal_task_ids.txt --workers 16 --model $MODEL "${AUTO[@]}"
