# CLAUDE.md

## Objective

MLSys 2026 Competition Track A solver. Given a DAG of `MatMul` and `Pointwise` ops over 2D tensors, produce a schedule that partitions ops into fused subgraphs with optimized tile granularity, minimizing total execution latency on a simulated hardware platform (roofline cost model: `step_latency = max(compute_time, memory_time)`).

Input: problem JSON (tensors, ops, hardware specs). Output: solution JSON (subgraphs with op assignments, granularity, retain decisions).

## Build & Run

```bash
make              # debug build (g++ -std=c++20 -g -O0)
make release      # optimized build (-O2 -DNDEBUG, static on Linux)
make submission   # release + zip package for submission
```

```bash
./build/mlsys <input.json> <output.json>
```

### Environment Variables

| Variable | Values | Default | Purpose |
|---|---|---|---|
| `MLSYS_SOLVER` | `baseline`, `optimus`, `optimus_conv`, `optimus_conv_v2`, `optimus_paper` | `optimus` | Solver backend |
| `MLSYS_OPTIMUS_CANDIDATES` | `interval`, `seed` | `interval` | Candidate generation mode |
| `MLSYS_OPTIMUS_CONV_DATAFLOW` | `os`, `ws`, `is`, `rs` | `rs` | Conv dataflow (conv backends only) |
| `MLSYS_OPTIMUS_MAX_SEG_SIZE` | integer | `25` | Max segment size for graph-cut decomposition (`optimus_paper` only) |
| `MLSYS_OPTIMUS_FRONTIER_MEMO_CAP` | integer | `10000000` | Frontier DP memo cap (states) |

Example:
```bash
MLSYS_SOLVER=optimus_conv_v2 MLSYS_OPTIMUS_CANDIDATES=seed ./build/mlsys benchmarks/mlsys-2026-1.json outputs/out.json
```

## Repo Structure

```
source/
  main.cpp                  — Entry point, CLI argument handling
  problem.h / problem.cpp   — JSON input parsing; Tensor, Op, Problem structs
  solution.h / solution.cpp — Solution/Subgraph structs, JSON serialization
  evaluator.h / evaluator.cpp — Independent latency scorer & validity checker
  solver.h / solver.cpp     — Backend dispatcher + baseline solver implementation
  optimus.h / optimus.cpp   — Core Optimus solver (candidate gen, DP partition, granularity search)
  conv_accelerator.h / conv_accelerator.cpp — Conv-guided cost model & scheduling heuristics
third_party/
  json.hpp                  — nlohmann/json single-header lib (download separately)
benchmarks/                 — Released benchmark JSON files (download separately)
outputs/                    — Generated solution JSON files
Makefile                    — Build system (debug / release / submission targets)
run_benchmarks.sh           — Script to run all released benchmarks
```

## Key Files

### `source/problem.h/.cpp`
Data structures: `Tensor` (width, height), `Op` (op_type, inputs, outputs, base_cost), `Granularity` (w, h, k), `Problem` (tensors, ops, hardware specs). `ReadProblem()` parses JSON input and derives producer/consumer graph edges and graph inputs/outputs.

### `source/solution.h/.cpp`
Data structures: `Subgraph` (op_ids, tensors_to_retain, granularity, traversal_order, subgraph_latency), `Solution` (vector of subgraphs). `WriteSolution()` serializes to JSON.

### `source/evaluator.h/.cpp` (~788 lines)
Independent scorer aligned with official competition rules. Builds internal `OpGraph` (producer/consumer, topological order, adjacency). Simulates tile-level execution with split-K steps, tracks memory footprint, validates fusion legality (connected sub-DAG, boundary tensors, OOM checks). Key functions: `Evaluate()`, `EvaluateSubgraph()`, `RecomputeLatencies()`.

### `source/solver.h/.cpp` (~351 lines)
Dispatches to backends via `MLSYS_SOLVER` env var. Contains the `baseline` backend (one op per subgraph, granularity enumeration, no fusion). Builds spatial/reduction dimension candidates and validates working set against `fast_memory_capacity`.

### `source/optimus.h/.cpp` (~2057 lines, largest file)
Core algorithm. Candidate generation (interval-based or seed-growth). DAG construction with `OpGraph`. Group boundary analysis (boundary inputs/outputs, internal tensors). Working set validation. DP partition search with memoization (`SearchStateKey`). Granularity selection with cached scoring. Conv-guided variants add penalty/reranking on top of the base search.

### `source/conv_accelerator.h/.cpp` (~459 lines combined)
Conv-specific structures (`Conv2DOp`, `ConvTileShape`, `ConvAcceleratorSpec`). Pseudo-conv mapping from MatMul/Pointwise to conv-like ops. `AnalyzeConvChain()` estimates buffer footprint and traffic. `DecideISOAForChain()` uses DP for in-situ activation reuse. `PropagateOutputTilesBackward()` propagates tile constraints through the op chain.

### `source/main.cpp` (~44 lines)
Entry point. Reads problem, calls `Solve()`, validates with `RecomputeLatencies()`, writes solution.

## Architecture

```
Solve()
├── baseline              — No fusion, one op per subgraph
└── optimus family
    ├── optimus           — Main solver (most stable, best empirical results)
    ├── optimus_conv      — Conv guidance v1 (global additive penalty)
    └── optimus_conv_v2   — Conv guidance v2 (local top-K reranking)
    Each supports:
    ├── interval candidates   — Topological consecutive ranges (default, stable)
    └── seed-growth candidates — DAG-aware expansion from seed ops (experimental)
```

- The **evaluator** is an independent scorer, not a rubber-stamp of solver-provided latencies. It re-simulates every subgraph tile-by-tile.
- The **optimus solver** calls `EvaluateSubgraph()` directly during search, so the solver's cost model is aligned with the final scorer.
- **Conv guidance** uses pseudo-conv mapping (tensor height → spatial, width → channels) since the problem schema only has MatMul/Pointwise, not real conv ops.

## Known Benchmark Results

### Best config: `optimus_paper` + `seed` (with graph-cut decomposition)

| Benchmark | N ops | Latency | Wall time |
|---|---:|---:|---:|
| mlsys-2026-1 | 5 | 405,875 | <1s |
| mlsys-2026-5 | 19 | 759,125 | <1s |
| mlsys-2026-9 | 32 | 164,506,000 | ~43s |
| mlsys-2026-13 | 63 | 166,404,000 | ~9s |
| mlsys-2026-17 | 103 | 46,338,000 | ~2s |

Run with:
```bash
MLSYS_SOLVER=optimus_paper MLSYS_OPTIMUS_CANDIDATES=seed ./build/mlsys benchmarks/mlsys-2026-N.json outputs/out-N.json
```

### Baseline: `optimus` + `interval`

| Benchmark | Latency |
|---|---:|
| mlsys-2026-1 | 405,875 |
| mlsys-2026-5 | 915,215 |
| mlsys-2026-9 | 164,506,000 |
| mlsys-2026-13 | 166,406,000 |
| mlsys-2026-17 | 5,013,690 |
