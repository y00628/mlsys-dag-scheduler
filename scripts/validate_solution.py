#!/usr/bin/env python3
"""Standalone structural validator for MLSys DAG scheduler solutions.

Backup validator — replicates the C++ evaluator's structural checks
(evaluator.cpp RecomputeImpl, lines 618-754) in pure Python. Use this
when you can't run the C++ binary (e.g. validating old solution JSONs
from git, or on a machine without a build environment).

The primary validation path is the C++ binary itself (main.cpp calls
RecomputeLatencies). compare_candidate_modes.py parses its stderr for
ERROR: lines.  This script is the offline fallback.

Usage:
    python3 scripts/validate_solution.py <problem.json> <solution.json>
    python3 scripts/validate_solution.py --batch <benchmarks_dir/> <solutions_dir/>
"""

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Tensor:
    width: int
    height: int


@dataclass
class Op:
    op_type: str
    inputs: List[int]
    outputs: List[int]
    base_cost: int


@dataclass
class Problem:
    tensors: List[Tensor]
    ops: List[Op]
    fast_memory_capacity: int
    slow_memory_bandwidth: int
    native_granularity: Tuple[int, int]
    graph_inputs: List[int]
    graph_outputs: List[int]


@dataclass
class OpGraph:
    producer_of_tensor: List[int]          # tensor_id -> op_id or -1
    consumers_of_tensor: List[List[int]]   # tensor_id -> [op_id, ...]
    preds: List[List[int]]                 # op_id -> [predecessor op_ids]
    succs: List[List[int]]                 # op_id -> [successor op_ids]
    topo_order: List[int]                  # ops in topological order
    topo_pos: List[int]                    # op_id -> position in topo_order


@dataclass
class BoundaryInfo:
    boundary_inputs: List[int]
    boundary_outputs: List[int]
    internal_tensors: List[int]


@dataclass
class Subgraph:
    op_ids: List[int]
    tensors_to_retain: List[int]
    granularity: Tuple[int, int, int]  # (width, height, depth)
    subgraph_latency: float


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Problem / solution parsing
# ---------------------------------------------------------------------------

def read_problem(path: str) -> Problem:
    """Parse problem JSON (mirrors problem.cpp ReadProblem)."""
    with open(path, "r") as f:
        j = json.load(f)

    widths = j["widths"]
    heights = j["heights"]
    num_tensors = len(widths)
    tensors = [Tensor(width=widths[i], height=heights[i]) for i in range(num_tensors)]

    op_types = j["op_types"]
    base_costs = j["base_costs"]
    inputs = j["inputs"]
    outputs = j["outputs"]
    num_ops = len(op_types)
    ops = [
        Op(
            op_type=op_types[i],
            inputs=list(inputs[i]),
            outputs=list(outputs[i]),
            base_cost=base_costs[i],
        )
        for i in range(num_ops)
    ]

    fast_memory_capacity = j["fast_memory_capacity"]
    slow_memory_bandwidth = j["slow_memory_bandwidth"]
    ng = j["native_granularity"]
    native_granularity = (ng[0], ng[1])

    # Derive graph inputs/outputs (same logic as problem.cpp lines 62-75)
    produced: Set[int] = set()
    consumed: Set[int] = set()
    for op in ops:
        produced.update(op.outputs)
        consumed.update(op.inputs)

    graph_inputs = [i for i in range(num_tensors) if i not in produced]
    graph_outputs = [i for i in range(num_tensors) if i not in consumed]

    return Problem(
        tensors=tensors,
        ops=ops,
        fast_memory_capacity=fast_memory_capacity,
        slow_memory_bandwidth=slow_memory_bandwidth,
        native_granularity=native_granularity,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
    )


def read_solution(path: str) -> List[Subgraph]:
    """Parse solution JSON (mirrors solution.cpp)."""
    with open(path, "r") as f:
        j = json.load(f)

    subgraphs_raw = j.get("subgraphs", [])
    retain_raw = j.get("tensors_to_retain", [])
    gran_raw = j.get("granularities", [])
    lat_raw = j.get("subgraph_latencies", [])

    n = len(subgraphs_raw)
    subgraphs: List[Subgraph] = []
    for i in range(n):
        op_ids = list(subgraphs_raw[i])
        tensors_to_retain = list(retain_raw[i]) if i < len(retain_raw) else []
        g = gran_raw[i] if i < len(gran_raw) else [1, 1, 1]
        granularity = (g[0], g[1], g[2])
        latency = lat_raw[i] if i < len(lat_raw) else 0.0
        subgraphs.append(Subgraph(
            op_ids=op_ids,
            tensors_to_retain=tensors_to_retain,
            granularity=granularity,
            subgraph_latency=latency,
        ))

    return subgraphs


# ---------------------------------------------------------------------------
# Graph construction (mirrors evaluator.cpp BuildOpGraph, lines 98-150)
# ---------------------------------------------------------------------------

def build_op_graph(problem: Problem) -> OpGraph:
    num_tensors = len(problem.tensors)
    num_ops = len(problem.ops)

    producer_of_tensor = [-1] * num_tensors
    consumers_of_tensor: List[List[int]] = [[] for _ in range(num_tensors)]

    for op_id, op in enumerate(problem.ops):
        for t in op.outputs:
            producer_of_tensor[t] = op_id
        for t in op.inputs:
            consumers_of_tensor[t].append(op_id)

    preds: List[List[int]] = [[] for _ in range(num_ops)]
    succs: List[List[int]] = [[] for _ in range(num_ops)]
    indegree = [0] * num_ops

    for op_id, op in enumerate(problem.ops):
        for t in op.inputs:
            producer = producer_of_tensor[t]
            if producer >= 0:
                preds[op_id].append(producer)
                succs[producer].append(op_id)
                indegree[op_id] += 1

    # Kahn's algorithm for topological sort
    q = deque(i for i in range(num_ops) if indegree[i] == 0)
    topo_order: List[int] = []
    while q:
        op_id = q.popleft()
        topo_order.append(op_id)
        for s in succs[op_id]:
            indegree[s] -= 1
            if indegree[s] == 0:
                q.append(s)

    topo_pos = [0] * num_ops
    for i, op_id in enumerate(topo_order):
        topo_pos[op_id] = i

    return OpGraph(
        producer_of_tensor=producer_of_tensor,
        consumers_of_tensor=consumers_of_tensor,
        preds=preds,
        succs=succs,
        topo_order=topo_order,
        topo_pos=topo_pos,
    )


# ---------------------------------------------------------------------------
# Boundary computation (mirrors evaluator.cpp ComputeBoundaryInfo, lines 215-255)
# ---------------------------------------------------------------------------

def compute_boundary_info(
    problem: Problem, graph: OpGraph, op_ids: List[int]
) -> BoundaryInfo:
    op_set = set(op_ids)
    boundary_inputs: Set[int] = set()
    boundary_outputs: Set[int] = set()
    internal_tensors: Set[int] = set()

    for op_id in op_ids:
        for t in problem.ops[op_id].inputs:
            producer = graph.producer_of_tensor[t]
            if producer < 0 or producer not in op_set:
                boundary_inputs.add(t)
            else:
                internal_tensors.add(t)

        for t in problem.ops[op_id].outputs:
            consumers = graph.consumers_of_tensor[t]
            escapes = len(consumers) == 0
            if not escapes:
                for c in consumers:
                    if c not in op_set:
                        escapes = True
                        break
            if escapes:
                boundary_outputs.add(t)

    return BoundaryInfo(
        boundary_inputs=sorted(boundary_inputs),
        boundary_outputs=sorted(boundary_outputs),
        internal_tensors=sorted(internal_tensors),
    )


# ---------------------------------------------------------------------------
# Connectivity check (mirrors evaluator.cpp IsConnectedSubgraph, lines 181-213)
# ---------------------------------------------------------------------------

def is_connected_subgraph(graph: OpGraph, op_ids: List[int]) -> bool:
    if len(op_ids) <= 1:
        return len(op_ids) == 1

    op_set = set(op_ids)
    visited: Set[int] = set()
    q = deque([op_ids[0]])
    visited.add(op_ids[0])

    while q:
        op_id = q.popleft()
        for nxt in graph.succs[op_id]:
            if nxt in op_set and nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
        for nxt in graph.preds[op_id]:
            if nxt in op_set and nxt not in visited:
                visited.add(nxt)
                q.append(nxt)

    return len(visited) == len(op_ids)


# ---------------------------------------------------------------------------
# Common output shape check (mirrors evaluator.cpp SharesCommonOutputShape, lines 162-179)
# ---------------------------------------------------------------------------

def shares_common_output_shape(problem: Problem, op_ids: List[int]) -> bool:
    if not op_ids:
        return False
    ref_op = problem.ops[op_ids[0]]
    if len(ref_op.outputs) != 1:
        return False
    ref_tensor = problem.tensors[ref_op.outputs[0]]

    for op_id in op_ids:
        op = problem.ops[op_id]
        if len(op.outputs) != 1:
            return False
        t = problem.tensors[op.outputs[0]]
        if t.width != ref_tensor.width or t.height != ref_tensor.height:
            return False
    return True


# ---------------------------------------------------------------------------
# Topological contiguity check
# ---------------------------------------------------------------------------

def is_contiguous_in_topo(op_ids: List[int], topo_pos: List[int]) -> bool:
    if not op_ids:
        return False
    positions = sorted(topo_pos[o] for o in op_ids)
    return positions[-1] - positions[0] + 1 == len(positions)


# ---------------------------------------------------------------------------
# Main validation (mirrors evaluator.cpp RecomputeImpl, lines 618-754)
# ---------------------------------------------------------------------------

def validate(
    problem: Problem, subgraphs: List[Subgraph]
) -> ValidationResult:
    result = ValidationResult()
    graph = build_op_graph(problem)
    num_ops = len(problem.ops)

    # Check 1: non-empty solution
    if not subgraphs:
        result.errors.append("Solution contains no subgraphs")
        return result

    covered_ops: Set[int] = set()
    slow_available: Set[int] = set(problem.graph_inputs)
    retained_prev: Set[int] = set()
    total_latency = 0.0

    for sg_idx, sg in enumerate(subgraphs):
        # Check 2: non-empty subgraph
        if not sg.op_ids:
            result.errors.append(f"Subgraph {sg_idx} is empty")
            continue

        # Check 3: positive granularity
        w, h, d = sg.granularity
        if w <= 0 or h <= 0 or d <= 0:
            result.errors.append(
                f"Subgraph {sg_idx} has non-positive granularity ({w}, {h}, {d})"
            )

        # Check 4: op range validity
        for op_id in sg.op_ids:
            if op_id < 0 or op_id >= num_ops:
                result.errors.append(
                    f"Subgraph {sg_idx} references invalid op {op_id}"
                )

        # Check 5: no duplicate ops within subgraph
        unique_ops = set(sg.op_ids)
        if len(unique_ops) != len(sg.op_ids):
            result.errors.append(
                f"Subgraph {sg_idx} has duplicate ops"
            )

        # Track coverage
        covered_ops.update(sg.op_ids)

        # Sort ops by topo order for subsequent checks
        valid_ops = [o for o in sg.op_ids if 0 <= o < num_ops]
        ordered_ops = sorted(valid_ops, key=lambda o: graph.topo_pos[o])

        if not ordered_ops:
            continue

        # Check 7: connected sub-DAG
        if not is_connected_subgraph(graph, ordered_ops):
            result.errors.append(
                f"Subgraph {sg_idx} is not a connected sub-DAG"
            )

        # Check 8: common output shape
        if not shares_common_output_shape(problem, ordered_ops):
            result.errors.append(
                f"Subgraph {sg_idx} mixes incompatible output shapes"
            )

        # Compute boundary info
        boundary = compute_boundary_info(problem, graph, ordered_ops)

        # Check 10: retain validity
        boundary_outputs_set = set(boundary.boundary_outputs)
        retained_now: Set[int] = set()
        for t in sg.tensors_to_retain:
            if t not in boundary_outputs_set:
                result.errors.append(
                    f"Subgraph {sg_idx} retains tensor {t} "
                    f"that is not one of its boundary outputs"
                )
            retained_now.add(t)

        # Check 9: tensor availability
        for t in boundary.boundary_inputs:
            if t not in retained_prev and t not in slow_available:
                result.errors.append(
                    f"Subgraph {sg_idx} needs tensor {t} "
                    f"that is unavailable in fast or slow memory"
                )

        # Update memory state (lines 727-732)
        for t in boundary.boundary_outputs:
            if t not in retained_now:
                slow_available.add(t)
        retained_prev = retained_now

        total_latency += sg.subgraph_latency

        # Warning: non-contiguous subgraph
        if len(ordered_ops) > 1 and not is_contiguous_in_topo(ordered_ops, graph.topo_pos):
            positions = sorted(graph.topo_pos[o] for o in ordered_ops)
            span = positions[-1] - positions[0] + 1
            result.warnings.append(
                f"Subgraph {sg_idx} is non-contiguous in topo order "
                f"(ops span {span} positions but only {len(ordered_ops)} ops)"
            )

    # Check 6: full op coverage
    missing_ops = [op_id for op_id in range(num_ops) if op_id not in covered_ops]
    if missing_ops:
        truncated = missing_ops[:20]
        suffix = f", ... ({len(missing_ops)} total)" if len(missing_ops) > 20 else ""
        result.errors.append(
            f"Op coverage: {len(missing_ops)} ops missing — "
            f"{truncated}{suffix}"
        )

    # Check 11: graph output availability
    for t in problem.graph_outputs:
        if t not in slow_available:
            result.errors.append(
                f"Graph output tensor {t} is not available in slow memory at program end"
            )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def validate_one(problem_path: str, solution_path: str) -> ValidationResult:
    problem = read_problem(problem_path)
    subgraphs = read_solution(solution_path)

    prob_name = os.path.basename(problem_path)
    sol_name = os.path.basename(solution_path)
    num_ops = len(problem.ops)
    num_tensors = len(problem.tensors)

    # Count ops covered
    all_ops: Set[int] = set()
    for sg in subgraphs:
        all_ops.update(sg.op_ids)

    print(f"=== {prob_name} <- {sol_name} ===")
    print(f"Problem: {num_tensors} tensors, {num_ops} ops")
    print(f"Solution: {len(subgraphs)} subgraphs, {len(all_ops)}/{num_ops} ops covered")

    result = validate(problem, subgraphs)

    if result.errors:
        print(f"\nERRORS:")
        for e in result.errors:
            print(f"  [FAIL] {e}")

    if result.warnings:
        print(f"\nWARNINGS:")
        for w in result.warnings:
            print(f"  [WARN] {w}")

    total_lat = sum(sg.subgraph_latency for sg in subgraphs)
    status = "VALID" if result.valid else "INVALID"
    nerr = len(result.errors)
    nwarn = len(result.warnings)
    print(f"\nTotal latency: {total_lat:.2f}")
    print(f"RESULT: {status} ({nerr} errors, {nwarn} warnings)")
    print()

    return result


def find_matching_problem(solution_path: str, benchmarks_dir: str) -> Optional[str]:
    """Try to find a matching problem file for a solution file.

    Matches by benchmark number: mlsys-2026-17_*.json -> mlsys-2026-17.json
    """
    sol_name = os.path.basename(solution_path)
    # Extract benchmark identifier (e.g. "mlsys-2026-17" from "mlsys-2026-17_seed_growth.json")
    # Try progressively shorter prefixes
    parts = sol_name.replace(".json", "").split("_")
    for end in range(len(parts), 0, -1):
        candidate = "_".join(parts[:end]) + ".json"
        candidate_path = os.path.join(benchmarks_dir, candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate MLSys DAG scheduler solution files against problem specifications."
    )
    parser.add_argument(
        "problem_or_benchdir",
        help="Problem JSON file, or benchmarks directory (with --batch)",
    )
    parser.add_argument(
        "solution_or_soldir",
        help="Solution JSON file, or solutions directory (with --batch)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: validate all solutions in a directory against matching benchmarks",
    )
    args = parser.parse_args()

    if args.batch:
        bench_dir = args.problem_or_benchdir
        sol_dir = args.solution_or_soldir

        if not os.path.isdir(bench_dir):
            print(f"Error: {bench_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(sol_dir):
            print(f"Error: {sol_dir} is not a directory", file=sys.stderr)
            sys.exit(1)

        sol_files = sorted(
            f for f in os.listdir(sol_dir) if f.endswith(".json")
        )
        if not sol_files:
            print(f"No .json files found in {sol_dir}", file=sys.stderr)
            sys.exit(1)

        total = 0
        passed = 0
        failed = 0
        skipped = 0

        for sol_file in sol_files:
            sol_path = os.path.join(sol_dir, sol_file)
            prob_path = find_matching_problem(sol_file, bench_dir)
            if prob_path is None:
                print(f"[SKIP] {sol_file}: no matching problem file in {bench_dir}")
                skipped += 1
                continue

            total += 1
            try:
                result = validate_one(prob_path, sol_path)
                if result.valid:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[ERROR] {sol_file}: {e}")
                failed += 1

        print(f"{'='*60}")
        print(f"BATCH SUMMARY: {passed}/{total} valid, {failed}/{total} invalid, {skipped} skipped")
        sys.exit(0 if failed == 0 else 1)

    else:
        problem_path = args.problem_or_benchdir
        solution_path = args.solution_or_soldir

        if not os.path.exists(problem_path):
            print(f"Error: {problem_path} not found", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(solution_path):
            print(f"Error: {solution_path} not found", file=sys.stderr)
            sys.exit(1)

        result = validate_one(problem_path, solution_path)
        sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
