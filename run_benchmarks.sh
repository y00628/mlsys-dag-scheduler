#!/bin/bash
# Run the solver against all available benchmarks.
#
# Defaults to the paper-Optimus backend with seed-growth candidates and the
# bitmask/frontier set-cover DP, which matches the best-validated config in
# this repo. Override with env:
#   MLSYS_SOLVER                    (default: optimus_paper)
#   MLSYS_OPTIMUS_CANDIDATES        (default: seed)
#   MLSYS_OPTIMUS_FRONTIER_MEMO_CAP (default: 10000000, used for N > 20)
#   MLSYS_VALIDATE=1                also run scripts/validate_solution.py
set -e

BINARY="./build/mlsys"
BENCH_DIR="benchmarks"
OUT_DIR="outputs"

: "${MLSYS_SOLVER:=optimus_paper}"
: "${MLSYS_OPTIMUS_CANDIDATES:=seed}"
: "${MLSYS_OPTIMUS_FRONTIER_MEMO_CAP:=10000000}"
export MLSYS_SOLVER MLSYS_OPTIMUS_CANDIDATES MLSYS_OPTIMUS_FRONTIER_MEMO_CAP

mkdir -p "$OUT_DIR"

if [ ! -f "$BINARY" ]; then
    echo "Binary not found. Run 'make' first."
    exit 1
fi

if [ ! -d "$BENCH_DIR" ]; then
    if [ -d "MLSys/benchmarks" ]; then
        BENCH_DIR="MLSys/benchmarks"
    else
        echo "Benchmark directory not found. Expected 'benchmarks/' or 'MLSys/benchmarks/'."
        exit 1
    fi
fi

echo "Backend: MLSYS_SOLVER=$MLSYS_SOLVER  MLSYS_OPTIMUS_CANDIDATES=$MLSYS_OPTIMUS_CANDIDATES"
echo "Frontier memo cap: $MLSYS_OPTIMUS_FRONTIER_MEMO_CAP"
echo ""

# Arrays for end-of-run summary.
declare -a SUMMARY_NAMES
declare -a SUMMARY_LATS
declare -a SUMMARY_ERRS
declare -a SUMMARY_SECS

# Sort benchmarks by the trailing numeric index so bm-1, bm-5, bm-9, ... come
# out in natural order instead of lexicographic order.
bench_files=$(ls "$BENCH_DIR"/*.json 2>/dev/null | \
              awk -F'[-.]' '{print ($(NF-1)+0)"\t"$0}' | sort -n | cut -f2)

for input in $bench_files; do
    name=$(basename "$input" .json)
    output="$OUT_DIR/${name}_solution.json"
    log="$OUT_DIR/${name}.log"
    echo "=== $name ==="

    start=$(date +%s.%N)
    "$BINARY" "$input" "$output" >"$log" 2>&1 || true
    end=$(date +%s.%N)
    secs=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')

    lat=$(grep "Total latency" "$log" | tail -1 | awk '{print $3}')
    errs=$(grep -c "^ERROR:" "$log" || true)

    # Echo the key lines so the live output stays informative.
    grep -E "Optimus paper|Solution:|Total latency|^ERROR:" "$log" || true
    echo "  wall=${secs}s"
    echo ""

    SUMMARY_NAMES+=("$name")
    SUMMARY_LATS+=("${lat:-n/a}")
    SUMMARY_ERRS+=("${errs:-0}")
    SUMMARY_SECS+=("$secs")
done

echo "=== Summary ==="
printf "%-24s %-16s %-8s %s\n" "benchmark" "latency" "errors" "wall_sec"
printf "%-24s %-16s %-8s %s\n" "------------------------" "----------------" "--------" "--------"
for i in "${!SUMMARY_NAMES[@]}"; do
    printf "%-24s %-16s %-8s %s\n" \
        "${SUMMARY_NAMES[$i]}" "${SUMMARY_LATS[$i]}" "${SUMMARY_ERRS[$i]}" "${SUMMARY_SECS[$i]}"
done

# Optional validation pass.
if [ "${MLSYS_VALIDATE:-0}" = "1" ] && [ -f "scripts/validate_solution.py" ]; then
    echo ""
    echo "=== Validating solutions ==="
    for i in "${!SUMMARY_NAMES[@]}"; do
        name="${SUMMARY_NAMES[$i]}"
        input="$BENCH_DIR/${name}.json"
        output="$OUT_DIR/${name}_solution.json"
        if [ -f "$output" ]; then
            result=$(python3 scripts/validate_solution.py "$input" "$output" 2>&1 | grep "^RESULT:" | tail -1)
            printf "%-24s %s\n" "$name" "${result:-(no RESULT line)}"
        fi
    done
fi

echo ""
echo "All benchmarks complete. Solutions in $OUT_DIR/, logs in $OUT_DIR/*.log"
