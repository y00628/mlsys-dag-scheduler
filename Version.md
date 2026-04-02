# Version

## Overview

目前 repo 內的 solver 共有四個 backend：

1. `baseline`
2. `optimus`
3. `optimus_conv`
4. `optimus_conv_v2`

其中 `optimus` 家族內部還有兩種 candidate generation mode：

- `interval`
- `seed-growth`

## Backend Tree

```text
Solve()
├─ baseline
└─ optimus family
   ├─ optimus
   │  ├─ interval candidates
   │  └─ seed-growth candidates
   ├─ optimus_conv
   │  ├─ interval candidates
   │  └─ seed-growth candidates
   └─ optimus_conv_v2
      ├─ interval candidates
      └─ seed-growth candidates
```

## How To Switch

### Backend

- `baseline`
  - `MLSYS_SOLVER=baseline`
- `optimus`
  - 預設，或 `MLSYS_SOLVER=optimus`
- `optimus_conv`
  - `MLSYS_SOLVER=optimus_conv`
- `optimus_conv_v2`
  - `MLSYS_SOLVER=optimus_conv_v2`

### Candidate Mode

- `interval`
  - 預設
- `seed-growth`
  - `MLSYS_OPTIMUS_CANDIDATES=seed`

範例：

```bash
MLSYS_SOLVER=optimus_conv_v2 MLSYS_OPTIMUS_CANDIDATES=seed ./build/mlsys <input.json> <output.json>
```

## Version Details

### 1. `baseline`

目的：

- 提供最基本、最容易理解的比較基準

方法：

- 每個 op 自己成一個 subgraph
- 為單一 op 枚舉合法 granularity
- 用同一套 contest latency 規則評估

特性：

- 沒有 fusion search
- 沒有 conv guidance
- 可和其他 backend 做公平比較

適用情境：

- sanity check
- regression comparison

### 2. `optimus`

目的：

- 作為目前主要的 contest-aligned backend

方法：

- 建立 producer-consumer DAG
- 產生 candidate groups
- 用 DP 選 partition
- 用 evaluator 重算 latency 與合法性

特性：

- 是目前主版本
- 已與 contest scorer 對齊
- 支援 `interval` 與 `seed-growth` candidate mode

目前狀態：

- 整體最穩定
- 目前 benchmark 表現最佳或與最佳持平

### 3. `optimus_conv`

目的：

- 將 `conv_accelerator` 第一次接進 active solver path
- 嘗試用 paper-like conv schedule heuristic 幫助 search

方法：

- 保留 contest evaluator 當最終 scorer
- 使用 pseudo-conv mapping 將 `MatMul` / `Pointwise` 映射到 conv-style chain
- 將 conv guidance 的 penalty 直接加到 proxy ranking 上

特性：

- 是第一版 conv-guided backend
- 但 guidance 權重較粗，容易干擾原本排序

目前狀態：

- 可正常編譯與執行
- 在部分 benchmark，尤其 `mlsys-2026-5`，會比 `optimus` 更差

### 4. `optimus_conv_v2`

目的：

- 修正 `optimus_conv` 直接全域加權導致排序變差的問題

方法：

- 不再用全域 additive penalty
- 先以 contest proxy 排序
- 再對 top-K 候選做 conv-guided local rerank
- 目前只用 normalized traffic ratio 做 rerank
- `parameter_refills` 與 `subgroup_count` 先只用於輕量 pruning

特性：

- 是目前較穩定的 conv-guided 版本
- 不容易像 `optimus_conv` 那樣破壞原本最佳解

目前狀態：

- 大多數 benchmark 可追平 `optimus`
- 目前還沒有穩定超過 `optimus`
- 大圖上 runtime 仍然是主要問題

## Candidate Modes

### `interval`

方法：

- 依 topological order 枚舉連續區間作為 candidate group

優點：

- 穩定
- 搜尋空間較小
- runtime 易控制

缺點：

- 搜尋空間較窄
- 離完整 DAG-aware fusion search 還有距離

### `seed-growth`

方法：

- 從 seed op 出發，沿 graph 擴張出 candidate

優點：

- 比 `interval` 更接近 DAG-aware search

缺點：

- runtime 較高
- 目前仍在實驗與調整中

## Current Recommendation

- 日常比較與主要結果：
  - 使用 `optimus`
- 若要看 conv-guided path：
  - 優先使用 `optimus_conv_v2`
- 若要當對照組：
  - 使用 `baseline`
- 若要做 search experiment：
  - 在 `optimus` 或 `optimus_conv_v2` 上切 `MLSYS_OPTIMUS_CANDIDATES=seed`

## Current Gap To Paper Backend

目前最接近 paper 的是：

- `optimus_conv_v2`

但它仍然不是完整 paper backend，因為還缺：

- 真正 conv operator schema
- 真正 accelerator config input
- 完整 DAG-aware fusion search
- 讓 conv scheduler 主導 search，而不只是 guidance
- 更完整的 ISOA / scheduler integration
