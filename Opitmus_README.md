# Optimus README

## Build

建議先用 release build：

```bash
make release
```

產生的 binary 在：

```bash
./build/mlsys
```

## Basic Usage

指令格式：

```bash
./build/mlsys <input.json> <output.json>
```

範例：

```bash
./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_solution.json
```

## Solver Backends

目前共有四個 backend：

1. `baseline`
2. `optimus`
3. `optimus_conv`
4. `optimus_conv_v2`

### 1. Default: `optimus`

不需要額外設定，直接跑：

```bash
./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_solution.json
```

或顯式指定：

```bash
MLSYS_SOLVER=optimus ./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_solution.json
```

### 2. `baseline`

```bash
MLSYS_SOLVER=baseline ./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_baseline.json
```

### 3. `optimus_conv`

```bash
MLSYS_SOLVER=optimus_conv ./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_optimus_conv.json
```

### 4. `optimus_conv_v2`

```bash
MLSYS_SOLVER=optimus_conv_v2 ./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_optimus_conv_v2.json
```

## Candidate Generation Modes

`optimus` family 內部支援兩種 candidate generation mode：

1. `interval`
2. `seed-growth`

### Default: `interval`

不需要額外設定。

### `seed-growth`

```bash
MLSYS_OPTIMUS_CANDIDATES=seed ./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_seed.json
```

### Backend + Seed Together

例如用 `optimus_conv_v2` 搭配 `seed-growth`：

```bash
MLSYS_SOLVER=optimus_conv_v2 MLSYS_OPTIMUS_CANDIDATES=seed ./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_conv_v2_seed.json
```

## Conv Dataflow

`optimus_conv` / `optimus_conv_v2` 支援切換 conv dataflow：

- `os`
- `ws`
- `is`
- `rs`

目前預設是：

```bash
rs
```

### Example

```bash
MLSYS_SOLVER=optimus_conv_v2 MLSYS_OPTIMUS_CONV_DATAFLOW=rs ./build/mlsys benchmarks/mlsys-2026-1.json outputs/mlsys-2026-1_conv_v2_rs.json
```

## Run All Released Benchmarks

```bash
./run_benchmarks.sh
```

## Common Command Summary

### Main version

```bash
./build/mlsys <input.json> <output.json>
```

### Baseline

```bash
MLSYS_SOLVER=baseline ./build/mlsys <input.json> <output.json>
```

### Conv v1

```bash
MLSYS_SOLVER=optimus_conv ./build/mlsys <input.json> <output.json>
```

### Conv v2

```bash
MLSYS_SOLVER=optimus_conv_v2 ./build/mlsys <input.json> <output.json>
```

### Seed mode

```bash
MLSYS_OPTIMUS_CANDIDATES=seed ./build/mlsys <input.json> <output.json>
```

### Conv v2 + Seed

```bash
MLSYS_SOLVER=optimus_conv_v2 MLSYS_OPTIMUS_CANDIDATES=seed ./build/mlsys <input.json> <output.json>
```

## Notes

- 預設 backend 是 `optimus`
- `optimus_conv` 是第一版 conv-guided backend
- `optimus_conv_v2` 是目前較穩定的 conv-guided backend
- 若只是一般使用或交叉比較，建議優先跑：
  - `optimus`
  - `baseline`
  - `optimus_conv_v2`
