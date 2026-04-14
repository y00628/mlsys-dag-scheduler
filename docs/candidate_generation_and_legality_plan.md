# Candidate Group Generation + Valid Fused Group Checking Plan

> Scope owner: Andrew  
> Focus modules: **Candidate Group Generation** / **Valid Fused Group Checking**

## 1) Scope from checklist

### Candidate Group Generation
- [x] Make **DAG-aware seed growth** the main path
- [x] Support **general DAG groups** (not only linear/contiguous patterns)

### Valid Fused Group Checking
- [ ] Move legality closer to paper-style **execution-validity**
- [ ] Reduce dependence on shape-based early reject heuristics

---

## 2) Design goals

1. Improve candidate search-space quality without runtime explosion.
2. Separate **hard legality constraints** from **heuristic policy**.
3. Keep outputs explainable (every reject has reason code).
4. Ensure downstream partition/cost modules can consume stable candidate schema.

---

## 3) Deliverable contracts (before coding)

## 3.1 GroupCandidate schema
Each emitted candidate should include:
- `op_ids` (canonical sorted ids)
- `seed_id` / growth metadata (optional but useful)
- `boundary_inputs`, `boundary_outputs`, `internal_tensors`
- `topo_footprint` (or equivalent ordering context)
- `internalized_bytes` (or estimated saved traffic)
- `debug_tags` (optional)

## 3.2 LegalityResult schema
- `is_valid: bool`
- `level_passed: {L0,L1,L2,L3}` (or equivalent)
- `reason_code` when invalid
- optional diagnostics: working-set estimate, violating op/tensor ids

Suggested reason codes:
- `DISCONNECTED_SUBGRAPH`
- `ORDERING_CONFLICT`
- `BOUNDARY_UNSATISFIED`
- `WORKING_SET_OOM`
- `HEURISTIC_REJECT_*`

---

## 4) Phase plan

## Phase 0 — Contract freeze (1–2 days)

### Tasks
- [x] Finalize `GroupCandidate` and `LegalityResult` fields
- [x] Define legality levels and reason code dictionary
- [x] Define runtime guard knobs (frontier budget, max group size, candidate budget)

### Implemented notes
- Added `LegalityLevel`, `LegalityReason`, `LegalityResult` in `source/optimus.cpp`.
- Extended `CandidateGroup` with phase-0 contract fields:
  - `seed_id`, `growth_depth`, `topo_footprint`, `legality`, `debug_tags`.
- Added reason-code string helpers for debugging/reporting.
- Added seed-growth runtime guard config parsing from env:
  - `MLSYS_OPTIMUS_SEED_MAX_GROUP`
  - `MLSYS_OPTIMUS_SEED_MAX_FRONTIER`
  - `MLSYS_OPTIMUS_SEED_MAX_CANDIDATES_PER_START`
  - `MLSYS_OPTIMUS_SEED_TOTAL_QUEUE_BUDGET`

### Exit criteria
- All downstream modules can consume schema without private assumptions.

---

## Phase 1 — Candidate generation upgrade (3–5 days)

### Tasks
- [x] Promote seed-growth to default candidate generation path
- [x] Implement expansion operators:
  - [x] forward successor expansion
  - [x] backward predecessor expansion
  - [x] bidirectional expansion for branch/diamond structures
- [ ] Maintain boundary/internal tensor sets incrementally during growth
  - blocker: boundary/internal tensor tracking is still recomputed per candidate path; no incremental delta-update state is wired yet.
- [x] Canonicalization + de-dup key
- [ ] Add dominance/rank-based pruning and hard runtime budgets
  - blocker: ranking/policy pruning and budgets exist, but seed-growth still times out on larger released benchmarks (e.g., 9/13/17 with 60s cap), so bounded-runtime criterion is not yet met.

### Implemented notes (current)
- Seed-growth is now default when `MLSYS_OPTIMUS_CANDIDATES` is unset.
- Added bidirectional frontier helper `CollectGrowthFrontier(...)` with predecessor/successor toggles.
- Removed contiguous-topology hard gate from seed-growth acceptance path.
- Added canonicalization/dedup helpers:
  - `CanonicalizeOpSet(...)`
  - `CandidateKeyFromOps(...)`
- Replaced `std::set<std::vector<size_t>> seen` with hash-based dedup over canonical op-sets.
- Runtime guard knobs are wired in seed-growth path:
  - `MLSYS_OPTIMUS_SEED_MAX_GROUP`
  - `MLSYS_OPTIMUS_SEED_MAX_FRONTIER`
  - `MLSYS_OPTIMUS_SEED_MAX_CANDIDATES_PER_START`
  - `MLSYS_OPTIMUS_SEED_TOTAL_QUEUE_BUDGET`

### Exit criteria
- Non-contiguous DAG candidates are generated consistently.
- Candidate count is bounded by guard rails on all released benchmarks.

### Phase 1 implementation task list (function-level)

> Target file: `source/optimus.cpp`  
> Goal: make seed-growth the default/high-quality path and support more general DAG groups.

#### P1-1. Seed-growth as default path
- [x] **Change default candidate mode** in `GetCandidateGenerationMode()`
  - from `kInterval` -> `kSeedGrowth` when env var is unset.
- [x] Keep interval fallback via env override (`MLSYS_OPTIMUS_CANDIDATES=interval`).

#### P1-2. Expand frontier beyond successor-only growth
- [x] Refactor frontier building in `GenerateSeedGrowthCandidates(...)`
  - currently frontier uses only `graph.succs[op_id]`
  - add predecessor frontier (`graph.preds[op_id]`) and/or bi-direction policy.
- [x] Add helper function:
  - [x] `CollectGrowthFrontier(const OpGraph&, const std::vector<size_t>& current_ops, bool allow_predecessor_growth, bool allow_successor_growth)`
  - output: sorted dedup candidate next ops.

#### P1-3. Remove contiguous-topology gate for DAG groups
- [x] In `GenerateSeedGrowthCandidates(...)`, remove hard gate:
  - `IsContiguousTopoSpan(graph, candidate)`
- [x] Replace with connectivity + legality-first gate:
  - retain `BuildBestCandidate(...)` validity checks
  - add/keep hard connectedness checks (`IsConnectedSubDAG(...)`).

#### P1-4. Canonicalization and dedup robustness
- [x] Replace `std::set<std::vector<size_t>> seen` usage with canonical key helper to reduce overhead.
- [x] Add helper function:
  - [x] `CanonicalizeOpSet(const std::vector<size_t>& ops, const OpGraph& graph)`
  - [x] `CandidateKeyFromOps(const std::vector<size_t>& canonical_ops)`
- [x] Keep final dedup by `candidate.ops`, but ensure only canonicalized forms enter queue.

#### P1-5. Runtime guards (hard budget knobs)
- [x] Add config/env knobs (read once near `GetCandidateGenerationMode()` region):
  - [x] `MLSYS_OPTIMUS_SEED_MAX_GROUP`
  - [x] `MLSYS_OPTIMUS_SEED_MAX_FRONTIER`
  - [x] `MLSYS_OPTIMUS_SEED_MAX_CANDIDATES_PER_START`
  - [x] `MLSYS_OPTIMUS_SEED_TOTAL_QUEUE_BUDGET`
- [x] Apply guards inside `GenerateSeedGrowthCandidates(...)`:
  - cap frontier expansion
  - cap queue growth
  - cap per-start emitted candidates.

#### P1-6. Quality filtering as policy layer (not hard legality)
- [x] Keep hard validity in `BuildBestCandidate(...)`:
  - `SharesCommonOutputShape(...)`, `IsConnectedSubDAG(...)`, scorer-valid metrics.
- [x] Move aggressive candidate suppression to policy phase in `GenerateSeedGrowthCandidates(...)`:
  - internalized-bytes threshold
  - density ranking/truncation (currently top-8).
- [x] Add a policy helper function:
  - [x] `PassSeedPolicyFilter(const CandidateGroup&, const Problem&)`.

Implemented:
- Added `PassSeedPolicyFilter(...)` and extracted internalized-threshold suppression out of core growth logic.
- Added `RankAndTrimSeedCandidates(...)` to centralize policy ranking + truncation.
- Added policy env knobs:
  - `MLSYS_OPTIMUS_SEED_POLICY_ENFORCE_INTERNALIZED`
  - `MLSYS_OPTIMUS_SEED_POLICY_MIN_INTERNALIZED`

#### P1-7. Better logging for candidate coverage debug
- [x] Extend `SeedDebugEnabled()` logs in `GenerateSeedGrowthCandidates(...)`:
  - [x] number of explored states
  - [x] number of accepted/rejected candidates
  - top reject reasons (if available from legality layer later).

#### P1-8. Integration checkpoint with solver path
- [x] Ensure `SolveWithOptimusImpl(...)` uses upgraded seed-growth path safely:
  - candidate pool non-empty at each start
  - single-op fallback still guaranteed.
- [x] Validate no regression in `BuildSolutionFromSearch(...)` and `SolveFromState(...)` assumptions:
  - `candidate.start`/`candidate.end` ordering remains valid
  - overlap and progression logic still terminates.

Implemented:
- Added first-pass validation script: `scripts/compare_candidate_modes.py`
- Script runs both `interval` and `seed_growth` under `optimus` backend and reports:
  - wall time
  - total latency
  - subgraph count
  - selected non-contiguous ratio
  - seed-start candidate stats (for seed mode)

#### P1-9. Minimal test/validation checklist for Phase 1
- [x] On released benchmarks, collect:
  - [x] candidate count per `start` (seed debug parsing)
  - [x] percentage of non-contiguous groups (selected schedule proxy)
  - [x] runtime (wall-clock proxy)
- [ ] Compare three modes:
  - [x] interval vs seed-growth(current)
  - [ ] seed-growth(old) baseline (requires historical commit / snapshot run)
    - blocker: old seed-growth numbers are not yet reproduced from a pinned historical commit/snapshot in the current workspace runbook.
- [ ] Pass criteria:
  - [ ] no empty-start candidate buckets
    - blocker: currently only verified on successful runs; full released-benchmark verification is incomplete because several seed-growth runs time out.
  - [x] no crash / no infinite queue growth (guarded by queue budget + successful runs)
  - [ ] candidate diversity improved with bounded runtime (pending benchmark report)
    - blocker: diversity improvement appears on completed subsets, but bounded-runtime is not satisfied across all released benchmarks due to seed-growth timeouts.

#### P1-10. Suggested commit slicing (for clean review)
- [ ] Commit A: mode default + env knobs
- [ ] Commit B: frontier refactor + bidirectional growth
- [ ] Commit C: contiguous gate removal + canonical dedup
- [ ] Commit D: runtime guards + policy filter helper
- [ ] Commit E: debug metrics + benchmark comparison notes

---

## Phase 2 — Legality checker refactor (3–5 days)

### Legality levels
- **L0 (Graph-level hard checks):** connectedness, no illegal cycles
- **L1 (Execution validity):** dependency/boundary satisfiable under execution order
- **L2 (Resource validity):** working-set/tile feasibility under memory constraints
- **L3 (Policy heuristics):** optional conservative rules (tunable/on-off)

### Tasks
- [ ] Implement layered legality pipeline (L0→L1→L2→L3)
- [ ] Convert existing shape rejects into L3 policy wherever possible
- [ ] Emit reason codes for every rejection
- [ ] Add debug trace hooks for reject analysis

### Exit criteria
- Hard invalid and policy reject are clearly separated.
- Turning off L3 still yields stable valid candidates.

---

## Phase 3 — Integration and stabilization (2–4 days)

### Tasks
- [ ] Integrate candidate generation + legality as one pipeline
- [ ] Ensure incremental legality updates during growth (avoid full recompute)
- [ ] Add config switches:
  - [ ] `candidate_mode=seed_growth|interval|hybrid`
  - [ ] legality strictness / heuristic on-off
  - [ ] runtime budgets

### Exit criteria
- End-to-end pipeline runs without regression in basic functionality.
- Runtime stays within budget envelope.

---

## Phase 4 — Validation/Ablation pack (2–3 days)

### Metrics to report
- Candidate coverage:
  - total candidates
  - unique non-interval candidate ratio
- Legality stats:
  - pass rate per level (L0/L1/L2/L3)
  - top reject reason codes
- Runtime stats:
  - generation time
  - legality time
  - peak candidate pool size

### Required ablations
- [ ] interval vs seed-growth vs hybrid
- [ ] with/without L3 heuristics
- [ ] budget sensitivity (frontier/group-size/candidate budget)

### Exit criteria
- Quantitative evidence that seed-growth path is better enough to be default.

---

## 5) Risks and mitigations

- **Risk:** candidate explosion  
  **Mitigation:** frontier caps + candidate budget + dominance pruning

- **Risk:** over-conservative legality kills good groups  
  **Mitigation:** strict separation of hard constraints vs policy heuristics

- **Risk:** debugging is opaque  
  **Mitigation:** reason-code-first design and per-candidate trace snippets

---

## 6) Definition of Done (DoD)

- [ ] DAG-aware seed-growth is main generation path
- [ ] General DAG (non-contiguous) fused groups are supported
- [ ] Legality checker is layered and explainable with reason codes
- [ ] Shape early reject is no longer the primary gate
- [ ] Validation report exists (coverage/quality/runtime/ablation)
- [ ] Feature flags allow fallback to interval mode for safety

---

## 7) Suggested branch naming

- `feature/seed-growth-legality-v1`
- `feature/candidate-legality-refactor`
- `andrew/candidate-validity-phase1`
