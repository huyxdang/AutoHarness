#!/usr/bin/env bash
# Qwen3.5-27B experiment: same protocol as the 9B run, with its own harness copy (harnesses/27b,
# starting from AppWorld's ReAct baseline) and its own history file (optimizer/history_27b.md).
#   1. Baseline ReAct on all 168 test_normal tasks   (runs alongside step 2; the GPU has room)
#   2. Optimization loop: round 0 on dev, then N=3 rounds (train -> Claude edits -> dev keep/reject)
#   3. Best harness on all 168 test_normal tasks
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=qwen3.5-27b

$PY scripts/run_eval.py --experiment 27b_test_react --dataset test_normal \
    --task-ids-file results/test_normal_task_ids.txt --model $MODEL \
    --agent-type simplified_react_code_agent > results_27b_test_react.log 2>&1 &
BASELINE_PID=$!

$PY optimizer/loop.py --rounds 3 --model $MODEL --tag 27b_ --harness-dir harnesses/27b > results_27b_loop.log 2>&1
echo "loop exit: $?"

$PY scripts/run_eval.py --experiment 27b_test_auto --dataset test_normal \
    --task-ids-file results/test_normal_task_ids.txt --model $MODEL \
    --agent-type autoharness_react_code_agent --prompt-file harnesses/27b/prompt.txt \
    --harness-dir harnesses/27b > results_27b_test_auto.log 2>&1
echo "final test exit: $?"

wait $BASELINE_PID
echo "baseline test exit: $?"
