#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RunResult:
    benchmark: str
    mode: str
    wall_time_sec: float
    total_latency: float
    n_subgraphs: int
    selected_non_contiguous_ratio: float
    seed_starts: int
    empty_seed_starts: int
    avg_candidates_per_start: float


def topo_order(problem: dict) -> List[int]:
    n = len(problem["op_types"])
    producer = {}
    for op_id, outs in enumerate(problem["outputs"]):
        for t in outs:
            producer[t] = op_id

    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    indeg = [0] * n
    for op_id, ins in enumerate(problem["inputs"]):
        for t in ins:
            p = producer.get(t, None)
            if p is None:
                continue
            preds[op_id].append(p)
            succs[p].append(op_id)
            indeg[op_id] += 1

    q = [i for i in range(n) if indeg[i] == 0]
    order = []
    while q:
        u = q.pop(0)
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order


def is_contiguous_in_topo(subgraph_ops: List[int], topo_pos: Dict[int, int]) -> bool:
    if not subgraph_ops:
        return False
    positions = sorted(topo_pos[o] for o in subgraph_ops)
    return positions[-1] - positions[0] + 1 == len(positions)


def parse_seed_stats(stderr: str) -> Tuple[int, int, float]:
    # Lines emitted when MLSYS_OPTIMUS_DEBUG_SEED=1:
    # seed start X candidates=Y
    matches = re.findall(r"seed start\s+(\d+)\s+candidates=(\d+)", stderr)
    if not matches:
        return 0, 0, 0.0
    counts = [int(c) for _, c in matches]
    seed_starts = len(counts)
    empty = sum(1 for c in counts if c == 0)
    avg = sum(counts) / max(1, seed_starts)
    return seed_starts, empty, avg


def run_one(binary: str, benchmark_path: str, mode: str, out_dir: str) -> RunResult:
    bench_name = os.path.splitext(os.path.basename(benchmark_path))[0]
    out_path = os.path.join(out_dir, f"{bench_name}_{mode}.json")

    env = os.environ.copy()
    env["MLSYS_SOLVER_BACKEND"] = "optimus"
    if mode == "interval":
        env["MLSYS_OPTIMUS_CANDIDATES"] = "interval"
        env["MLSYS_OPTIMUS_DEBUG_SEED"] = "0"
    else:
        env["MLSYS_OPTIMUS_CANDIDATES"] = "seed_growth"
        env["MLSYS_OPTIMUS_DEBUG_SEED"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.run(
        [binary, benchmark_path, out_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    t1 = time.perf_counter()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Run failed for {bench_name} [{mode}]\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    with open(out_path, "r", encoding="utf-8") as f:
        sol = json.load(f)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        problem = json.load(f)

    order = topo_order(problem)
    topo_pos = {op: i for i, op in enumerate(order)}

    subgraphs = sol.get("subgraphs", [])
    non_contiguous = sum(1 for sg in subgraphs if not is_contiguous_in_topo(sg, topo_pos))
    ratio = non_contiguous / max(1, len(subgraphs))

    seed_starts, empty_seed_starts, avg_candidates = parse_seed_stats(proc.stderr)

    return RunResult(
        benchmark=bench_name,
        mode=mode,
        wall_time_sec=t1 - t0,
        total_latency=float(sum(sol.get("subgraph_latencies", []))),
        n_subgraphs=len(subgraphs),
        selected_non_contiguous_ratio=ratio,
        seed_starts=seed_starts,
        empty_seed_starts=empty_seed_starts,
        avg_candidates_per_start=avg_candidates,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare optimus candidate modes (interval vs seed-growth) on released benchmarks."
    )
    parser.add_argument(
        "--binary",
        default="./build/mlsys",
        help="Path to solver binary (default: ./build/mlsys)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=sorted(glob.glob("MLSys/benchmarks/*.json")),
        help="Benchmark JSON paths (default: all MLSys/benchmarks/*.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/mode_compare",
        help="Output directory for generated solutions",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    modes = ["interval", "seed_growth"]
    results: List[RunResult] = []
    for bench in args.benchmarks:
        for mode in modes:
            results.append(run_one(args.binary, bench, mode, args.out_dir))

    print("\n# Mode comparison summary\n")
    print(
        "benchmark\tmode\twall_sec\ttotal_latency\tsubgraphs\tnon_contig_ratio\t"
        "seed_starts\tempty_starts\tavg_candidates"
    )
    for r in results:
        print(
            f"{r.benchmark}\t{r.mode}\t{r.wall_time_sec:.4f}\t{r.total_latency:.1f}\t"
            f"{r.n_subgraphs}\t{r.selected_non_contiguous_ratio:.3f}\t"
            f"{r.seed_starts}\t{r.empty_seed_starts}\t{r.avg_candidates_per_start:.2f}"
        )

    # Aggregate by mode
    print("\n# Aggregate by mode\n")
    for mode in modes:
        subset = [r for r in results if r.mode == mode]
        avg_wall = sum(r.wall_time_sec for r in subset) / max(1, len(subset))
        avg_lat = sum(r.total_latency for r in subset) / max(1, len(subset))
        avg_non_contig = sum(r.selected_non_contiguous_ratio for r in subset) / max(1, len(subset))
        avg_empty = sum(r.empty_seed_starts for r in subset) / max(1, len(subset))
        print(
            f"{mode}: avg_wall={avg_wall:.4f}s, avg_latency={avg_lat:.1f}, "
            f"avg_non_contig_ratio={avg_non_contig:.3f}, avg_empty_seed_starts={avg_empty:.2f}"
        )


if __name__ == "__main__":
    main()
