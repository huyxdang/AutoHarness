# Auto-Harness

Automated harness optimization for small open models on AppWorld. Full plan, decisions, and
resource links: see [docs/notes.md](docs/notes.md) — read it at the start of every session.

## Rules
- Git commits: do NOT add Claude as co-author (no `Co-Authored-By` lines).
- Report **pass@1** (AppWorld TGC, one attempt per task) as the headline metric, plus SGC, input tokens/task, time/task.
- Modal: use profile `dangxhwee2003` (`MODAL_PROFILE=dangxhwee2003`; switched from `hellgod67` on 2026-09-29) — keep GPU idle time near zero.
- Never tune on test_normal. Optimizer edits harness code/prompt only.
