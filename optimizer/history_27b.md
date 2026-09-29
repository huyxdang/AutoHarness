
## qwen3.5-27b — round 0 (starting harness)
Dev: pass@1 20.0%, tests 74.6%, 92,739 input tok/task, 13.6 steps/task

### Round 1 — KEPT (better on dev and within token budget)
- Edit: 8 of 10 failed tasks failed the `assert answers match … == 'null'` test because the agent passed a spurious `answer` (often a status summary like "Money sent back to Robert") to `complete_task` on action tasks whose expected answer is null — with 6 of those failing on that test alone — so I rewrote section D of the prompt to make the agent decide answer-vs-no-answer up front, provide `answer` only for explicit questions, and never pass a confirmation/summary/count as the answer on action/state-change tasks.
- Dev after edit: pass@1 55.0%, tests 81.6%, 107,650 input tok/task, 13.8 steps/task
- Best dev so far: pass@1 55.0%, tests 81.6%, 107,650 input tok/task, 13.8 steps/task

### Round 2 — KEPT (better on dev and within token budget)
- Edit: 1 of the 5 failed train tasks (d0b1f43_1) showed the repeated-failing-call pattern — it re-ran a byte-identical failing code block 11 times to the 50-step cap, burning 827k input tokens (5× average) and never calling complete_task — so I added a loop guard in react_agent.py that detects consecutive identical code submissions, injects a course-correction notice urging a different approach, and terminates the episode after 3 repeats.
- Dev after edit: pass@1 70.0%, tests 88.6%, 100,576 input tok/task, 14.1 steps/task
- Best dev so far: pass@1 70.0%, tests 88.6%, 100,576 input tok/task, 14.1 steps/task
