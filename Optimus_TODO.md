# Optimus TODO

## Paper Checklist

這份 checklist 的目的是整理：

- 如果要朝 `Optimus` paper 的方向做
- 一個完整 backend 需要完成哪些工作
- 目前 repo 哪些已做、哪些未做、哪些只做到部分

狀態標記：

- `[Done]`
- `[Partial]`
- `[Todo]`

## 1. Graph Construction

- `[Done]` 建立 model graph 的 DAG 表示
  - 將 op 當節點
  - 將 tensor dependency 當邊
  - 建 producer / consumer graph

- `[Done]` 建立 topological order

- `[Done]` 建立 predecessor / successor adjacency

## 2. Candidate Group Generation

- `[Partial]` 產生 fusion candidates
  - 已有 `interval` mode
  - 已有 `seed-growth` mode
  - 但目前主要穩定路徑仍偏 topo-interval

- `[Todo]` 讓 DAG-aware seed growth 成為主路徑

- `[Todo]` 支援更一般的 DAG group，而不只偏 linear / contiguous pattern

## 3. Valid Fused Group Checking

- `[Done]` 檢查 connected subgraph

- `[Done]` 分析 boundary inputs / boundary outputs / internal tensors

- `[Partial]` 檢查 fusion legality
  - 已有基本 connected / dependency 檢查
  - 但仍有偏保守的 heuristic 限制

- `[Todo]` 讓 legality 更接近 paper 的 execution-validity 定義

- `[Todo]` 降低對 shape-based early reject 的依賴

## 4. Partition Search

- `[Done]` 使用 DP 選擇 candidate partition

- `[Done]` 使用 memoization 避免重算

- `[Partial]` retain-aware state 已接入
  - 但整體 search quality 仍受 candidate 空間限制

- `[Todo]` 讓 partition search 更完整反映 paper 的 DAG fusion search

## 5. Cost Model

- `[Done]` 有 contest-aligned latency scorer
  - 由 `evaluator.cpp` 重算 latency / legality

- `[Partial]` 有 conv-style accelerator guidance model
  - 由 `conv_accelerator.cpp` 提供 working set / traffic / refill / ISOA 分析
  - 但目前只作為 guidance

- `[Todo]` 建立更完整的 paper-style memory cost model
  - activation traffic
  - parameter traffic
  - parameter reload
  - subgroup overhead

- `[Todo]` 讓 conv cost model 不只是 rerank，而能主導 candidate quality

## 6. Working Set Analysis

- `[Done]` contest path 有 working set 檢查

- `[Partial]` conv path 有 chain working set 分析

- `[Todo]` 讓 fused group 的 working set 分析更接近 paper accelerator model

- `[Todo]` 明確整合 activation / parameter / line buffer footprint

## 7. Tile Selection

- `[Done]` 有 granularity / tile candidate 枚舉

- `[Partial]` conv path 有 terminal tile 到 per-op tile 的 backward propagation

- `[Todo]` 讓 tile selection 更 strongly guided by accelerator model

- `[Todo]` 讓 tile search 與 scheduler / ISOA 一起最佳化

## 8. Scheduler

- `[Partial]` 已有 scheduler-like 分析骨架
  - subgroup_count
  - memory traffic estimate
  - pseudo conv tile analysis

- `[Todo]` 建立真正的 fused-group scheduler 決策流程

- `[Todo]` 讓 scheduler 成為 solver 主決策的一部分，而不是只做 ranking signal

## 9. ISOA

- `[Done]` 已有 ISOA analysis module
  - `DecideISOAForChain(...)`

- `[Partial]` ISOA 已存在於 `conv_accelerator`，但尚未真正接進 solver 主流程

- `[Todo]` 讓 solver 對同一 group 比較 ISOA / non-ISOA schedule

- `[Todo]` 讓 ISOA 影響 candidate cost 與最終決策

## 10. Accelerator Config

- `[Partial]` 已有 `ConvAcceleratorSpec`

- `[Partial]` 目前 accelerator spec 主要由 heuristic 推導

- `[Todo]` 將 accelerator spec 變成正式可配置輸入
  - dataflow
  - PE shape
  - throughput
  - line buffer
  - RF / SRAM capacity

## 11. Operator Schema

- `[Partial]` 已有 pseudo-conv mapping
  - 將 `MatMul` / `Pointwise` 映射為 conv-like chain

- `[Todo]` 建立正式 conv-oriented internal IR

- `[Todo]` 真正支援：
  - kernel size
  - stride
  - padding
  - channels
  - weights

## 12. Heuristic Pruning

- `[Done]` 已有 candidate pruning

- `[Done]` 已有 group size heuristic

- `[Partial]` 已有 conv-guided local rerank / pruning

- `[Todo]` 加強 runtime guard
  - rerank budget
  - chain size gate
  - internalized-bytes gate
  - conv guidance cache

- `[Todo]` 發展 online-style fast variant

## 13. Conv Backend Integration

- `[Done]` `conv_accelerator` 已編進 build

- `[Done]` 已有 active conv-guided backend
  - `optimus_conv`
  - `optimus_conv_v2`

- `[Partial]` conv backend 目前是 paper-like guidance path

- `[Todo]` 讓 conv backend 成為真正獨立的 paper backend

- `[Todo]` 讓 conv scheduler 主導 search，而不只是輔助 contest proxy

## 14. Evaluation / Validation

- `[Done]` 有獨立 evaluator，會重算 legality 與 latency

- `[Done]` benchmark comparison framework 已可用於比較 backend

- `[Todo]` 建立更完整的 ablation
  - interval vs seed-growth
  - optimus vs optimus_conv vs optimus_conv_v2
  - conv rerank budget / pruning gate
  - dataflow 設定差異

## Recommended Next Order

若目標是逐步逼近 paper backend，建議順序如下：

1. `[Todo]` 先收斂 `optimus_conv_v2` runtime
2. `[Todo]` 讓 conv guidance 從 rerank 變成 candidate filter / pruning 主力
3. `[Todo]` 讓 seed-growth 成為主要 candidate generation 路徑
4. `[Todo]` 放寬目前過度保守的合法性限制
5. `[Todo]` 讓 ISOA 真正接進 solver 主流程
6. `[Todo]` 建立正式 conv IR
7. `[Todo]` 補正式 accelerator config input
8. `[Todo]` 讓 conv scheduler 主導 backend

## Current Summary

- 目前最穩定的主版本是 `optimus`
- 目前最接近 paper 方向的是 `optimus_conv_v2`
- 目前最大的缺口不是 correctness，而是：
  - conv backend fidelity
  - search space quality
  - runtime scalability
