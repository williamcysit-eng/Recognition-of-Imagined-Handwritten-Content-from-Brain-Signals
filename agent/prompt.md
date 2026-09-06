Read `agent/state.json`, `agent/history.jsonl`, and the relevant project code before acting.

Perform exactly ONE architecture experiment intended to improve the objective metric named in state. Keep development-set selection strictly separate from the held-out test split.
An experiment is accepted only if its aggregate metric and every configured seed improve over the current best.

Workflow:

1. Use the `architect` task agent to choose one concrete architecture hypothesis and decompose it into the smallest ordered, independently valid commit-sized steps. It must account for accepted ideas, rejected ideas, recent failures, remaining budget, and existing project conventions.
2. Give the complete brief to the `integrity-auditor` task agent. It must independently audit the truthfulness of the claims, the proposed data path, split isolation, and resistance to metric gaming. If it returns `BLOCK`, send its exact findings to the architect for a revised brief, then audit the revision again. Do not start implementation until it returns `PASS`.
3. For each approved step, in order:
   - Use the `worker` task agent to implement only that step.
   - Use the `validator` task agent to review the actual diff and run cheap relevant validation at medium reasoning. If it blocks, send only its concrete findings back to the worker, then validate the correction again.
   - After validation passes, use the `committer` task agent to commit exactly that small architectural change. Record the returned commit hash before starting the next step.
4. After the last planned commit, use the `validator` once more to validate the complete experiment against the architect's hypothesis and invariants. If it blocks, append the minimal correction as another worker → validator → committer step, never amend an earlier commit, then repeat complete-experiment validation.
5. Give the approved brief, complete commit range, actual diff, data-path code, and cheap-validation evidence to the `integrity-auditor` for a final independent audit. If it returns `BLOCK`, send its findings to the architect, have the architect define the minimal correction, implement it through worker → validator → committer, then repeat both complete-experiment validation and the final integrity audit. Do not write the report until it returns `PASS`.
6. Write `agent/experiment.json` with this exact shape:

```json
{
  "idea": "short unique description of the architecture experiment",
  "summary": "what changed and why it could improve the objective",
  "cheap_validation": ["exact command or scenario and observed result"],
  "integrity_audit": {
    "status": "PASS",
    "design_evidence": ["specific code, split, or data-provenance evidence checked before implementation"],
    "final_evidence": ["specific commit, diff, training-path, or validation evidence checked after implementation"],
    "residual_risk": "remaining uncertainty, or 'none identified' with a concrete basis"
  },
  "commits": ["full Git commit hash for each small architectural change, in order"]
}
```

Then exit. The external controller will run the fixed evaluator and will either retain the complete candidate commit chain or reset the branch to the previous best commit.

Hard boundaries:

- Only the `committer` task agent may stage or commit files. No role may reset, restore, clean, checkout, switch, merge, rebase, amend, or rewrite Git history.
- Every version-controlled architecture change must be committed before exit. Remain on the starting branch and leave the index and worktree clean.
- The `integrity-auditor` is read-only. Its design and final `PASS` decisions are required; summarize its concrete evidence faithfully in the report.
- Do not modify `../loop.py`, `scripts/evaluate.py`, `scripts/train.sh`, `agent/prompt.md`, `agent/state.json`, `agent/history.jsonl`, `.omp/config.yml`, `.omp/WATCHDOG.yml`, or `.omp/agents/`.
- Do not read or evaluate on the held-out test partition.
- Do not perform hyperparameter-only tuning; change model architecture or architecture composition.
- Do not run indefinitely, perform more than one experiment, or begin follow-up work after writing the report.
- Do not claim an objective improvement. Only the controller-owned evaluator determines that.
