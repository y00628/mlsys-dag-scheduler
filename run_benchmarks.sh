#!/bin/bash
# Run the solver against all available benchmarks.
set -e

BINARY="./build/mlsys"
BENCH_DIR="benchmarks"
OUT_DIR="outputs"

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

for input in "$BENCH_DIR"/*.json; do
    name=$(basename "$input" .json)
    output="$OUT_DIR/${name}_solution.json"
    echo "=== $name ==="
    time "$BINARY" "$input" "$output"
    echo ""
done

echo "All benchmarks complete. Solutions in $OUT_DIR/"
