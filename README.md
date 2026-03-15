# MLSys 2026 Competition — Track A Solver

DAG scheduling optimizer for the MLSys 2026 Competition Track.

## Setup

1. Download [nlohmann/json](https://github.com/nlohmann/json/releases) single header and place it at:
   ```
   third_party/json.hpp
   ```

2. Download the released benchmarks from the [competition repo](https://github.com/yarongmu-google/MLSys/tree/main/benchmarks) into `benchmarks/`.

## Build

```bash
make          # debug build
make release  # optimized + static linked (submission binary)
```

## Run

```bash
./mlsys benchmarks/mlsys-2026-1.json output.json
```

## Project Structure

```
source/
  main.cpp          — Entry point, CLI argument handling
  problem.h/.cpp    — JSON parsing, Problem data structures
  solution.h/.cpp   — Solution data structures, JSON output
  evaluator.h/.cpp  — Latency evaluation (mirrors official scorer)
  solver.h/.cpp     — Scheduling algorithm (your main work goes here)
third_party/
  json.hpp          — nlohmann/json (download separately)
benchmarks/         — Released benchmark JSON files
Makefile            — Build system
```

## Submission

```bash
make submission
# Creates: TeamName_TrackA_1.zip
```

## License

Apache-2.0
