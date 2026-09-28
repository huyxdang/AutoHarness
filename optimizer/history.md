# Harness optimization history

Each round: an edit proposed by the optimizer from train-set failures, then kept or rejected
based on a fixed 20-task dev set (pass@1, tiebreak: unit-test pass rate; input tokens may rise at most 20%).
