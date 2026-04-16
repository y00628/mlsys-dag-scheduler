#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import time
import tempfile
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
    timed_out: bool
    valid: bool = True
    ops_covered: int = 0
    ops_total: int = 0
    validation_errors: int = 0


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


def benchmark_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", name)
    idx = int(m.group(1)) if m else 10**9
    return (idx, name)


def terminate_process_tree(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_one(binary: str, benchmark_path: str, mode: str, out_dir: str, timeout_sec: float) -> RunResult:
    bench_name = os.path.splitext(os.path.basename(benchmark_path))[0]
    out_path = os.path.join(out_dir, f"{bench_name}_{mode}.json")

    env = os.environ.copy()
    env["MLSYS_SOLVER_BACKEND"] = "optimus"
    if mode == "interval":
        env["MLSYS_OPTIMUS_CANDIDATES"] = "interval"
        env["MLSYS_OPTIMUS_DEBUG_SEED"] = "0"
    elif mode == "seed_growth_old":
        env["MLSYS_OPTIMUS_CANDIDATES"] = "seed_growth"
        env["MLSYS_OPTIMUS_DEBUG_SEED"] = "1"
        env["MLSYS_OPTIMUS_SEED_ALLOW_PRED"] = "0"
        env["MLSYS_OPTIMUS_SEED_ALLOW_SUCC"] = "1"
        env["MLSYS_OPTIMUS_SEED_REQUIRE_CONTIGUOUS"] = "1"
    else:
        env["MLSYS_OPTIMUS_CANDIDATES"] = "seed_growth"
        env["MLSYS_OPTIMUS_DEBUG_SEED"] = "1"
        env["MLSYS_OPTIMUS_SEED_ALLOW_PRED"] = "1"
        env["MLSYS_OPTIMUS_SEED_ALLOW_SUCC"] = "1"
        env["MLSYS_OPTIMUS_SEED_REQUIRE_CONTIGUOUS"] = "0"

    t0 = time.perf_counter()
    timed_out = False
    stderr_text = ""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as errf:
        err_path = errf.name

    with open(err_path, "w", encoding="utf-8") as err_stream:
        p = subprocess.Popen(
            [binary, benchmark_path, out_path],
            stdout=subprocess.DEVNULL,
            stderr=err_stream,
            text=True,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_sec
        while True:
            rc = p.poll()
            if rc is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                terminate_process_tree(p)
                break
            time.sleep(0.1)

    with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
        stderr_text = f.read()
    try:
        os.remove(err_path)
    except OSError:
        pass

    if timed_out:
        raise RuntimeError(
            f"Run timed out for {bench_name} [{mode}] after {timeout_sec}s\n"
            f"stderr(partial tail):\n{stderr_text[-4000:]}\n"
        )

    t1 = time.perf_counter()
    # Don't raise on non-zero return code — main.cpp now returns 1 for
    # invalid solutions (after writing the output file).  The ERROR: lines
    # in stderr are parsed below and recorded in the RunResult instead.

    with open(out_path, "r", encoding="utf-8") as f:
        sol = json.load(f)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        problem = json.load(f)

    order = topo_order(problem)
    topo_pos = {op: i for i, op in enumerate(order)}

    subgraphs = sol.get("subgraphs", [])
    non_contiguous = sum(1 for sg in subgraphs if not is_contiguous_in_topo(sg, topo_pos))
    ratio = non_contiguous / max(1, len(subgraphs))

    seed_starts, empty_seed_starts, avg_candidates = parse_seed_stats(stderr_text)

    # Validation: the C++ evaluator (RecomputeLatencies in main.cpp) prints
    # "ERROR:" lines to stderr for invalid solutions.  Parse them directly
    # instead of re-implementing the checks in Python.
    error_lines = [ln for ln in stderr_text.splitlines() if ln.startswith("ERROR:")]
    is_valid = len(error_lines) == 0

    # Count ops covered from the solution JSON for diagnostics.
    all_op_ids = {op for sg in subgraphs for op in sg}
    num_ops = len(problem.get("op_types", []))

    if not is_valid:
        print(f"  [INVALID] {bench_name} {mode}: {len(error_lines)} errors", flush=True)
        for err in error_lines[:5]:
            print(f"    {err}", flush=True)
        if len(error_lines) > 5:
            print(f"    ... and {len(error_lines) - 5} more", flush=True)

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
        timed_out=timed_out,
        valid=is_valid,
        ops_covered=len(all_op_ids),
        ops_total=num_ops,
        validation_errors=len(error_lines),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare optimus candidate modes (interval vs seed-growth-old vs seed-growth) on released benchmarks."
    )
    parser.add_argument(
        "--binary",
        default="./build/mlsys",
        help="Path to solver binary (default: ./build/mlsys)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=sorted(glob.glob("MLSys/benchmarks/*.json"), key=benchmark_sort_key),
        help="Benchmark JSON paths (default: all MLSys/benchmarks/*.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/mode_compare",
        help="Output directory for generated solutions",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=90.0,
        help="Per-run timeout in seconds (default: 90)",
    )
    parser.add_argument(
        "--max-benchmarks",
        type=int,
        default=0,
        help="Limit number of benchmarks (0 means all)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs when one run fails/times out",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    modes = ["interval", "seed_growth_old", "seed_growth"]
    benches = args.benchmarks
    if args.max_benchmarks > 0:
        benches = benches[: args.max_benchmarks]

    results: List[RunResult] = []
    for bench in benches:
        for mode in modes:
            print(f"[run] {os.path.basename(bench)} mode={mode}", flush=True)
            try:
                results.append(run_one(args.binary, bench, mode, args.out_dir, args.timeout_sec))
            except Exception as e:
                if args.continue_on_error:
                    print(f"[warn] {e}", flush=True)
                    continue
                raise

    # ---- Pretty-printed results table ----
    def fmt_latency(lat: float) -> str:
        if lat >= 1e6:
            return f"{lat/1e6:.2f}M"
        if lat >= 1e3:
            return f"{lat/1e3:.1f}K"
        return f"{lat:.1f}"

    print("\n# Mode comparison summary\n")
    headers = ["benchmark", "mode", "valid", "ops", "subgraphs", "latency", "wall_sec"]
    rows = []
    for r in results:
        status = "OK" if r.valid else f"FAIL({r.validation_errors})"
        rows.append([
            r.benchmark,
            r.mode,
            status,
            f"{r.ops_covered}/{r.ops_total}",
            str(r.n_subgraphs),
            fmt_latency(r.total_latency),
            f"{r.wall_time_sec:.2f}s",
        ])

    # Compute column widths for alignment.
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))

    # ---- Aggregate by mode ----
    print("\n# Aggregate by mode\n")
    agg_headers = ["mode", "valid", "avg_latency", "avg_wall", "runs"]
    agg_rows = []
    for mode in modes:
        subset = [r for r in results if r.mode == mode]
        if not subset:
            continue
        n_valid = sum(1 for r in subset if r.valid)
        avg_wall = sum(r.wall_time_sec for r in subset) / max(1, len(subset))
        avg_lat = sum(r.total_latency for r in subset) / max(1, len(subset))
        valid_str = f"{n_valid}/{len(subset)}"
        if n_valid < len(subset):
            valid_str += " !!!"
        agg_rows.append([
            mode,
            valid_str,
            fmt_latency(avg_lat),
            f"{avg_wall:.2f}s",
            f"{len(subset)}/{len(benches)}",
        ])

    agg_widths = [len(h) for h in agg_headers]
    for row in agg_rows:
        for i, cell in enumerate(row):
            agg_widths[i] = max(agg_widths[i], len(cell))

    print("  ".join(h.ljust(agg_widths[i]) for i, h in enumerate(agg_headers)))
    print("  ".join("-" * w for w in agg_widths))
    for row in agg_rows:
        print("  ".join(cell.ljust(agg_widths[i]) for i, cell in enumerate(row)))

    # Pass/fail hints for Phase-1 checklist.
    by_mode = {m: [r for r in results if r.mode == m] for m in modes}
    interval_complete = len(by_mode["interval"]) == len(benches) and len(benches) > 0
    seed_complete = len(by_mode["seed_growth"]) == len(benches) and len(benches) > 0
    interval_non_contig = sum(r.selected_non_contiguous_ratio for r in by_mode["interval"]) / max(1, len(by_mode["interval"]))
    seed_non_contig = sum(r.selected_non_contiguous_ratio for r in by_mode["seed_growth"]) / max(1, len(by_mode["seed_growth"]))
    seed_empty_ok = seed_complete and all(r.empty_seed_starts == 0 for r in by_mode["seed_growth"])

    print("\n# Phase-1 checklist hints\n")
    if not seed_complete:
        print(
            "no_empty_seed_starts(seed_growth): INCONCLUSIVE "
            f"(successful_runs={len(by_mode['seed_growth'])}/{len(benches)})"
        )
    else:
        print(f"no_empty_seed_starts(seed_growth): {'PASS' if seed_empty_ok else 'FAIL'}")

    if not (seed_complete and interval_complete):
        print(
            "candidate_diversity_improved(seed_growth vs interval): INCONCLUSIVE "
            f"(seed_runs={len(by_mode['seed_growth'])}/{len(benches)}, "
            f"interval_runs={len(by_mode['interval'])}/{len(benches)})"
        )
    else:
        print(
            "candidate_diversity_improved(seed_growth vs interval): "
            f"{'PASS' if seed_non_contig > interval_non_contig else 'FAIL'} "
            f"(seed={seed_non_contig:.3f}, interval={interval_non_contig:.3f})"
        )


if __name__ == "__main__":
    main()
