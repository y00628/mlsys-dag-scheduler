#!/bin/bash
# Run the solver against all available benchmarks.
set -e

BINARY="./mlsys"
BENCH_DIR="benchmarks"
OUT_DIR="outputs"

mkdir -p "$OUT_DIR"

if [ ! -f "$BINARY" ]; then
    echo "Binary not found. Run 'make' first."
    exit 1
fi

for input in "$BENCH_DIR"/*.json; do
    name=$(basename "$input" .json)
    output="$OUT_DIR/${name}_solution.json"
    echo "=== $name ==="
    time "$BINARY" "$input" "$output"
    echo ""
done

echo "All benchmarks complete. Solutions in $OUT_DIR/"
