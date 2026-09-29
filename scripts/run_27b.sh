#!/usr/bin/env bash
# Qwen3.5-27B experiment: same protocol as the 9B run, with its own harness copy (harnesses/27b,
# starting from AppWorld's ReAct baseline) and its own history file (optimizer/history_27b.md).
#   1. Baseline ReAct on all 168 test_normal tasks   (runs alongside step 2; the GPU has room)
#   2. Optimization loop: round 0 on dev, then N=3 rounds (train -> Claude edits -> dev keep/reject)
#   3. Best harness on all 168 test_normal tasks
# Safe to rerun after an interruption: finished tasks are kept (AUTOHARNESS_RESUME, see run_eval.py).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=qwen3.5-27b
export AUTOHARNESS_RESUME=1

# The H100 batches ~16 requests at once and queues the rest, so more tasks in flight only add
# billed waiting. The loop is the critical path: it gets 20 (a whole dev round at once).
AUTOHARNESS_MAX_CONTAINERS=12 $PY scripts/run_eval.py --experiment 27b_test_react --dataset test_normal \
    --task-ids-file results/test_normal_task_ids.txt --model $MODEL \
    --agent-type simplified_react_code_agent > results_27b_test_react.log 2>&1 &
BASELINE_PID=$!

if AUTOHARNESS_MAX_CONTAINERS=20 $PY optimizer/loop.py --rounds 3 --model $MODEL --tag 27b_ \
        --harness-dir harnesses/27b > results_27b_loop.log 2>&1; then
    echo "loop exit: 0"
    AUTOHARNESS_MAX_CONTAINERS=32 $PY scripts/run_eval.py --experiment 27b_test_auto --dataset test_normal \
        --task-ids-file results/test_normal_task_ids.txt --model $MODEL \
        --agent-type autoharness_react_code_agent --prompt-file harnesses/27b/prompt.txt \
        --harness-dir harnesses/27b > results_27b_test_auto.log 2>&1
    echo "final test exit: $?"
else
    # A failed round can leave an unscored edit in harnesses/27b, so never test it blindly.
    echo "loop failed; skipping the final test"
fi

wait $BASELINE_PID
echo "baseline test exit: $?"
