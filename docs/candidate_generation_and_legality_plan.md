# Candidate Group Generation + Valid Fused Group Checking Plan

> Scope owner: Andrew  
> Focus modules: **Candidate Group Generation** / **Valid Fused Group Checking**

## 1) Scope from checklist

### Candidate Group Generation
- [ ] Make **DAG-aware seed growth** the main path
- [ ] Support **general DAG groups** (not only linear/contiguous patterns)

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
- [ ] Finalize `GroupCandidate` and `LegalityResult` fields
- [ ] Define legality levels and reason code dictionary
- [ ] Define runtime guard knobs (frontier budget, max group size, candidate budget)

### Exit criteria
- All downstream modules can consume schema without private assumptions.

---

## Phase 1 — Candidate generation upgrade (3–5 days)

### Tasks
- [ ] Promote seed-growth to default candidate generation path
- [ ] Implement expansion operators:
  - [ ] forward successor expansion
  - [ ] backward predecessor expansion
  - [ ] bidirectional expansion for branch/diamond structures
- [ ] Maintain boundary/internal tensor sets incrementally during growth
- [ ] Canonicalization + de-dup key
- [ ] Add dominance/rank-based pruning and hard runtime budgets

### Exit criteria
- Non-contiguous DAG candidates are generated consistently.
- Candidate count is bounded by guard rails on all released benchmarks.

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
