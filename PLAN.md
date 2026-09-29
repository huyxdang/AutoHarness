# Auto-Harness — Project Plan & Resources

Last updated: 2026-09-28. Living document — update as decisions change.

## 1. Purpose

Portfolio project for an application to **Adaption Labs** (harness engineering).

**Pitch:** Adaption showed a better harness took a 27B open model from 67.10% → 85.92% on the
Harvey Legal Agent Benchmark with no weight changes — but the harness was designed by hand from
training-set trajectories. This project **automates that loop**: a coding agent reads a small
model's failed trajectories, rewrites the harness, and keeps only changes that improve held-out
results.

**Research question:** Can an automatically optimized harness make a *small open model* beat
standard hand-built harnesses on a public agent benchmark, at equal or lower token cost?

**Novelty angle:** Existing auto-harness papers (A-Evolve, Adaptive Auto-Harness, GEPA,
Meta-Harness, Continual Harness, SkillOS) optimize *frontier* solvers. Small open models are
untested — and that is exactly Adaption's thesis. "Auto-harness" alone is not novel; lead with
the small-model + enterprise framing.

**Product vision (future work, not built):** enterprise customer gives task description +
requirements + a few examples → system generates a preview benchmark (task + environment +
grader) → customer approves/gives feedback → full benchmark generated (Adaption's Adaptive
Data / "Invent a dataset") → train/held-out split → automated harness loop → report.
Fits Adaption's lineup: Adaptive Data (data) + AutoScientist (training) + **Auto-Harness**.

## 2. Key decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Benchmark | **AppWorld** | Credible (public leaderboard, used by A-Evolve), official splits, free local grading via unit tests (no LLM judge, no user simulator) |
| Not Harvey LAB | rejected | No official split (Adaption's 140 held-out tasks unpublished), needs paid dual LLM judges (Sonnet 4.6 + GPT-5.5), huge document contexts |
| Not τ²-bench | rejected | Needs a paid LLM user simulator |
| Not synthetic HR SQL task | rejected | No reputation / not credible |
| Harness blocks library | deferred to v2 | Automate the whole loop first; later derive blocks from the optimizer's accepted-edit log |
| Optimizer | **Claude Code headless (`claude -p`)** | Runs on existing subscription, no API bill |
| Solver serving | **SGLang on Modal** | Mac (M2, 16 GB) too small for long agent runs |
| Metric reported | **pass@1** (= AppWorld TGC, one attempt per task) + SGC + input tokens/task + time/task | |
| Adaption's Prime Agent / OpenCode numbers | NOT reused | They're on Harvey LAB, different tasks/judge/model — not comparable. Re-run harnesses ourselves on AppWorld |

## 3. Setup

### Benchmark: AppWorld (750 tasks)
| Split | Tasks | Role |
|---|---|---|
| train | 105 | Optimizer reads failed trajectories here |
| dev | 60 | Accept/reject each harness edit (score only) |
| test_normal | 168 | Final headline, run ONCE per harness |
| test_challenge | 417 | Out of scope (optional) |

Why dev ≠ test: any split used to make keep/reject decisions gets inflated (selection on noise).
AppWorld's rules also forbid tuning on test.

### Models
| Role | Model | Where |
|---|---|---|
| Solver (MVP) | Qwen3.5-4B (thinking disabled) | SGLang on Modal, 1×A10 |
| Solver (main) | Qwen3.5-9B | SGLang on Modal (L40S) — decide after MVP |
| Optimizer | Claude Code headless, **Claude Opus 4.8 (1M context), `claude-opus-4-8[1m]`** — one `claude -p` call per round does diagnosis + edit | user's subscription |
| AppWorld + harness + grading | — | local Mac |

The optimizer model is pinned in `optimizer/round.py` (`--optimizer-model`). Rounds run on 2026-09-28
(4B MVP round 1, 9B rounds 1–3) used the same model via the global default `opus[1m]`.

Qwen3.5 (Feb 2026) is the newest small Qwen family (0.8B/2B/4B/9B); Qwen3.6/3.8 only ship 27B+.

### Setup notes (learned during MVP)
- Server: `serve/sglang_server.py`, endpoint `https://hellgod67--autoharness-sglang-sglang.us-east.modal.direct/v1`,
  served model name `qwen3.5-4b`. API key in `.env` (`SGLANG_API_KEY`) and Modal secret `autoharness-sglang`.
  Cold start ≈ 2.3 min; returns 503 while starting (AppWorld's client retries). Scales to zero after 5 idle min.
- Thinking is disabled by prepending `{%- set enable_thinking = false %}` to the chat template, because
  AppWorld's client rejects extra request params (no `chat_template_kwargs`).
- AppWorld must be installed **from source** (PyPI 0.1.3 is too old for the agents package). The repo's
  Git LFS quota fails silently; the 4 `.bundle` files were fetched from media.githubusercontent.com and
  sha256-checked against the LFS pointers, then `appworld install --repo && appworld download data`.
  `appworld verify tasks` passes 147/147.
- Runner: `scripts/run_eval.py` (parallel processes, timing, pass@1, tokens from `usage.json`, steps).
  It patches `reasoning_content: null` → `""` (SGLang vs AppWorld incompatibility) for all harnesses.
- Model call settings for ALL harnesses: `max_tokens=1500`, `stop=["```\n"]` (as in AppWorld's CI ReAct
  config). Without them Qwen3.5-4B sometimes rambles to the 32k context limit (~10 min/call), stalling runs.
- Qwen3.5 has no 8B; upgrade path is Qwen3.5-9B (~18 GB bf16 → needs L40S, or FP8 on A10).
- Harness: `harness/react_agent.py` (copy of AppWorld's simplified ReAct agent, registered as
  `autoharness_react_code_agent`) + `harness/prompt.txt`. Optimizer: `optimizer/round.py`.

### Compute / budget
- Modal profile to use: **`hellgod67`** (~$20 credits left as of 2026-09-28). The globally active
  profile is a different one — run Modal commands with `MODAL_PROFILE=hellgod67`.
- Keep costs down: scale-to-zero when idle, cache weights in a Modal Volume, run 8–16 AppWorld
  tasks in parallel against one server.
- Local machine: Apple M2, 16 GB RAM, Python 3.11/3.12 + uv installed, Ollama installed (local fallback).

## 4. Comparisons (model fixed, only harness changes)

| Harness | Model | Priority |
|---|---|---|
| ReAct (AppWorld standard `simplified_react_code_agent`) | same small model | **Core baseline** |
| Auto-harness (ours) | same small model | **Core result** |
| Prime Agent | same small model | Next (via AppWorld MCP server) |
| OpenCode | same small model | If time (via AppWorld MCP server) |
| Leaderboard refs (cited, not run) | Opus 4.6 91.1, Qwen3-14B 86.9, gpt-oss-20b 76.2, Qwen2.5-32B/LOOP 72.6, GPT-4o/ReAct 48.8, Llama3-70B 8.9–24.4 (test_normal TGC) | context |

Leaderboard evidence for thesis: same Llama3-70B goes 8.9 (PlanExec) → 24.4 (FullCodeRefl) just by
harness. Note some small-model leaderboard entries likely include fine-tuning (e.g. LOOP = RL).

## 5. Optimization loop

1. Start from AppWorld's ReAct harness (agent code + prompt).
2. Run solver on train tasks → save trajectories + pass/fail.
3. `claude -p` reads failed trajectories → names recurring failure patterns → edits harness code/prompt → one-line rationale → git commit.
4. Run on dev → keep if pass@1 up AND input tokens not up more than ~20%; else revert.
5. Repeat 3–5 rounds.
6. Final harness run once on test_normal.

**Guardrails**
- Optimizer may edit harness code/prompt only — never the grader, AppWorld, data, or dev/test trajectories.
- No task-specific hardcoding (AppWorld rule).
- Temperature 0; rerun important comparisons; don't trust 1–2 point gains.

## 6. MVP (prove the pipeline end to end)

| Step | Done when |
|---|---|
| 1. SGLang + Qwen3-4B on Modal | test prompt answered from the Mac |
| 2. AppWorld installed + data downloaded | `appworld verify` passes |
| 3. ReAct + Qwen3-4B on 5 dev tasks | pass@1, **input tokens/task**, and **time/task** recorded |
| 4. One optimizer round: 10 train tasks' failures → 1 edit | git commit with rationale |
| 5. Re-run the 5 dev tasks | before/after comparison (even if 0 gain) |

**Go/no-go:** baseline ≈0% → move to Qwen3-8B; >~5 min/task → cut task counts / add parallelism;
works → scale to full loop.

## 7. Phases
| Phase | What | Est. |
|---|---|---|
| 0 | MVP | 1–2 h |
| 1 | Full loop + ReAct baseline + final test run | 2–3 h |
| 2 | Prime Agent + OpenCode baselines | 2–4 h each |
| 3 | Write-up | 1–2 h |

## 8. Deliverables
- Repo (harness, loop, Modal server)
- Results table: pass@1, SGC, input tokens/task, time/task per harness on test_normal
- Harness changelog (each edit, reason, kept/rejected) — the most interesting artifact
- Short write-up incl. product vision (phase-1 benchmark generation) as future work

## 9. Out of scope (v2+)
- Benchmark generation from customer spec
- Harness blocks library (derive from changelog)
- Post-training on successful trajectories (hand off to AutoScientist) — Adaption's 85.92 → 88.03 step

## 10. Resources

### Adaption Labs
- Harness post: https://adaptionlabs.ai/blog/a-better-harness-can-unlock-smaller-models
  - 67.10 → 85.92% criterion pass (harness only), → 88.03% after post-training (27B Qwen)
  - Fixes: bounded file reads, actionable tool errors, preserve key state in compaction, validate deliverables before exit
  - "training-set trajectories" → evaluated on "140 held-out tasks with no overlap"; split method, judge, and task IDs NOT published
  - vs Prime Agent 0.7.1 / OpenCode 1.18.23 on a smaller subset (size unstated): 84.13% (0.79 pts below Prime Agent) with 8.06M vs 17.48M input tokens (-54%); OpenCode 6.38M tokens, 75.40% (not independently verified)
- Careers listing: https://adaptionlabs.ai/careers?ashby_jid=ed60d392-f14d-4bdc-87c0-c96906f728e5 (Senior Research Scientist — check for MTS / research engineer roles)
- Adaptive Data docs: https://docs.adaptionlabs.ai/ — ingest → map columns → adapt → expand → evaluate → export; API + Python SDK
- Invent a dataset: https://docs.adaptionlabs.ai/adaptive-data/invent-a-dataset/ (`dataset_prompt`, `rows`, `training_type` = instruction_dataset | preference_pairs, `estimate=True`)
- Blueprint: https://adaptionlabs.ai/blog/blueprint

### Benchmark
- AppWorld repo: https://github.com/StonyBrookNLP/appworld
  - ReAct agent: `experiments/code/simplified/react_code_agent.py`
  - Prompt: `experiments/prompts/react_code_agent/instructions.txt`
  - Config: `experiments/configs/ci/react_code_agent.jsonnet` (temp 0, max_steps 40, stop ```` ```\n ````)
  - Self-hosted model: `model_config.base_url` templated with `MODEL_SERVER_URL` env var
  - Tokens per task: `usage_tracker.py` → `usage.json` per task (time/task not recorded — add a timer)
  - Other agents: `full_code_agent.py`, `function_calling_agent.py`
  - MCP server: `appworld serve mcp` (for Prime Agent / OpenCode)
  - Install: `pip install appworld && appworld install && appworld download data && pip install -e "experiments[simplified]"` (Python ≥3.11)
  - Evaluate: `appworld evaluate {experiment_name} {dataset}`
- AppWorld paper: https://arxiv.org/abs/2407.18901
- Leaderboard: https://appworld.dev/appworld/leaderboard (raw: https://appworld.dev/appworld/leaderboard.json)

### Related papers
- A-Evolve (2602.00359): https://arxiv.org/abs/2602.00359 — AppWorld, 50 train tasks for evolution / 50 test_normal for eval; frontier solvers; Sonnet 4.5 evolver; baselines Vanilla, APE, AWM
  - Code: https://github.com/A-EVO-Lab/a-evolve — AppWorld code on branch `main-release-0223`, `agentic-evolution/examples/` (`run_appworld_claude.py`, `appworld_agent_system_prompt.py`, saved Sonnet train_50 outputs). `main` has GEPA, Meta-Harness implementations.
- Adaptive Auto-Harness (2606.01770): https://arxiv.org/abs/2606.01770 — streams (PolyBench, CTF-Dojo, FutureX), no held-out split, very expensive (233M input tokens on one run); baselines A-Evolve, GEPA, Meta-Harness, Continual Harness, SkillOS
- Harvey LAB: https://github.com/harveyai/harvey-labs (~1,660 tasks, no official split, dual LLM judges)

### Comparison harnesses
- Prime Agent: https://github.com/PrimeIntellect-ai/prime-agent (MIT; JSON/RPC headless modes; self-improving via `/refine`; custom endpoint support unverified)
- OpenCode: https://github.com/anomalyco/opencode (MIT; moved from sst/opencode; custom OpenAI-compatible providers)

### Serving
- Modal SGLang example: https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/sglang_low_latency.py (built for 2×H100 — adapt to 1 smaller GPU, weight caching, scale-to-zero)

## 11. Open items
- User mentioned "you can also use this GitHub repo" but no link was included — ask.
- Confirm Modal GPU pricing vs $20 budget before scaling past MVP.

## 12. Results (2026-09-28)

Solver Qwen3.5-9B (SGLang, L40S), optimizer Claude Opus 4.8, 3 rounds, 15 train tasks/round, fixed 20-task dev set.

| Round | Edit | Dev pass@1 | Decision |
|---|---|---|---|
| 0 | AppWorld ReAct baseline | 15% | start |
| 1 | No `answer` on action tasks (complete_task rule) | 45% | kept |
| 2 | Concrete pagination recipe | 45% (fewer unit tests) | rejected |
| 3 | Exact-schema-key / filter sanity checks | 20% | rejected |

Dev noise check (same 20 tasks, rerun): ReAct 15% / 5%, auto harness 45% / 35%.

**test_normal (168 tasks, run once each):**

| Harness | pass@1 | SGC | Unit tests | Input tok/task | Steps | Sec/task |
|---|---|---|---|---|---|---|
| AppWorld ReAct (baseline) | 19.6% (33/168) | 7.1% | 67.4% | 171k | 19.2 | 205 |
| Auto harness (round 1) | **49.4% (83/168)** | 25.0% | 75.2% | 166k | 18.5 | 164 |

Paired: 58 tasks solved only by auto, 8 only by ReAct, 25 by both; exact McNemar p ≈ 1.8e-10.
Leaderboard context (test_normal TGC): GPT-4o + ReAct 48.8, Llama3-70B + ReAct 20.8 (2024 entries).

**External harness comparison (test_normal, 168 tasks, Qwen3.5-9B, 2026-09-29):**

| Harness | pass@1 | SGC | Unit tests | Input tok/task | Steps |
|---|---|---|---|---|---|
| AppWorld ReAct | 19.6% | 7.1% | 67.4% | 171k | 19.2 |
| OpenCode 1.18.33 | 22.6% (38/168) | 8.9% | 57.3% | 299k | 14.9 |
| AutoHarness (round 1) | **49.4%** | **25.0%** | **75.2%** | **166k** | 18.5 |

AutoHarness vs OpenCode: 59 only-auto vs 14 only-OpenCode, McNemar p ≈ 1e-7. OpenCode vs ReAct: p = 0.59.
OpenCode setup: AppWorld MCP tools filtered to AppWorld's official API predictor output (≤20 APIs,
`scripts/mcp_filter_proxy.py`), official function-calling prompt without demos, 64k-context server
(`qwen3.5-9b-64k`), shell/edit/web tools disabled for safety, 20-min cap (1 timeout, enforced by hand
after the in-process timer lagged). 23 tasks sent a >32k request; OpenCode solved 1 of them, so the
larger context did not drive the comparison. Prime Agent dropped (unsandboxed REPL; out of scope).

**OpenCode audit + v2 re-run (2026-09-29):** the first OpenCode run (22.6%) was handicapped by the adapter:
(1) predicted-APIs-only access left a needed API missing in ~half the tasks (162 unavailable-tool attempts);
(2) OpenCode's read-only host filesystem tools stayed on and the model searched the Mac for AppWorld's
file_system app in 21 tasks (0 solved). Checked and fine: tool-call parsing (0 raw tool-call leaks), task
clock, grading. v2 adapter: predicted APIs as direct tools + `api_docs__show_api_descriptions`,
`api_docs__show_api_doc`, `call_api` (any API on demand), host FS tools off, temperature 0, wall-clock
20-min cap killing the process group, shuffled order + start deadline (budget guard).
v2 on a random 99-task subset of test_normal: OpenCode 29.3% vs v1 22.2%, ReAct 17.2%, AutoHarness 48.5%
on the same tasks. AutoHarness vs OpenCode v2: 28 vs 9 discordant, p = 0.0026; OpenCode v2 vs ReAct p = 0.029.
Budget: after this run roughly $1–2 of Modal credit remains (estimate from GPU time; check the dashboard).
Completion run (2026-09-29 morning): 45 more tasks before the user stopped it → OpenCode v2 on 144/168 tasks:
ReAct 18.8%, OpenCode v1 20.8%, OpenCode v2 29.2%, AutoHarness 49.3% on the same 144. AutoHarness vs OpenCode v2:
42 vs 13 discordant, +20.1 pts (95% CI +11.1 to +29.9), p ≈ 1e-4. 24 tasks not run
(`results/9b_test_opencode_v2_not_run_ids.txt`); parts merged with `scripts/merge_summaries.py`.

**Modal account switch (2026-09-29):** project moved from workspace `hellgod67` (credits used up) to
`dangxhwee2003`. Pins: `MODAL_WORKSPACE` in `scripts/run_eval.py` (endpoint URLs are built from it),
`MODAL_PROFILE` default in `scripts/modal_eval.py`, CLAUDE.md. Secret `autoharness-sglang` recreated with the
same key. Deployed so far in the new workspace: `autoharness-sglang-9b-64k`. Same model revision, GPU type
(L40S) and server settings, so results stay comparable across the two accounts.
