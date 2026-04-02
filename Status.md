# Status

## 目前 `optimus` 與 paper 的關係

### 結論

目前 repo 內的 `optimus` 不是 paper 的原版實作，而是 **Optimus-inspired backend**。

它有借用 paper 的核心方向：

- 用 DAG-aware 的方式做 fusion search，而不是只看單純線性 chain
- 先判斷 candidate group 是否合法，再用 cost-guided 的方式選 partition
- 用 DP 來挑一組整體 schedule
- 用 heuristic 剪枝限制搜尋空間

但它和 paper 還有明顯差異：

- paper 是 CNN / conv accelerator 導向；本題資料模型只有 `MatMul` / `Pointwise`
- paper 的 cost model 依賴 conv-specific 資訊；目前 repo 不具備那些輸入欄位
- 目前 candidate group 還被限制成 topo-order 上的連續區間
- 目前 group 還要求 common output shape
- retain 目前是在 partition 做完後才另外選，不是直接放進 DP state 一起最佳化
- traversal order 目前沒有納入搜尋，輸出預設是 raster / `null`

### 現在 `optimus` 已做到的事

- 建 producer / consumer graph
- 檢查 connected sub-DAG
- 分析 boundary inputs / boundary outputs / internal tensors
- 枚舉部分 granularity 候選 `[w, h, k]`
- 用 heuristic 方式限制 group size
- 用 DP 在候選 group 上選 schedule
- 事後補 retain tensors

### 現在 `optimus` 還沒做到的事

- 完整重現 paper 的 search space
- 完整重現 paper 的 accelerator-specific cost model
- 把 retain decision 一起放進 partition DP
- 把 traversal order 一起放進搜尋
- 直接以 evaluator 的完整 recomputation logic 當作 solver 的目標函數

## Latency: 舊版、現在版、官方定義

### 官方定義

官方定義在 `/tmp/MLSys/PROBLEM.md`，核心規則如下：

- 每個 subgraph 需要指定一組 `granularity = [w, h, k]`
- 若 `w` / `h` 小於 tensor size，會形成多個 spatial tiles
- 若 MatMul 的 `k` 小於完整 reduction dimension，會形成多個 split-k steps
- `subgraph_latency` 必須是所有 tiles 與 split-k steps 的 latency 總和
- 每個 step 的 latency 使用 roofline model：
  - `step_latency = max(compute_time, memory_time)`
- 若 working set 超過 `fast_memory_capacity`，該 schedule invalid / OOM
- subgraph 內的 intermediate tensors 是 ephemeral：
  - 不占 fast memory
  - 不產生 slow-memory traffic
- `tensors_to_retain` 只控制 subgraph 間 residency
- `traversal_order` 會影響 tile 執行順序與資料重用

### 舊版 latency 是什麼

早期版本其實沒有真正的獨立 scorer。

當時的情況是：

- solver 端會自己填 `subgraph_latency`
- baseline 甚至接近 placeholder / 簡化估算
- evaluator 只把 solver 填進去的 `subgraph_latency` 加總

因此舊版數字的問題是：

- 不一定真的符合官方規則
- 不一定真的合法
- 沒有獨立檢查 retain / traversal / residency / OOM
- 數字通常偏小，因為很多 memory movement 與 execution detail 沒有真的算進去

### 現在版 latency 是什麼

目前的 `source/evaluator.cpp` 已改成 **獨立 recomputation + validity checker**。

它現在會：

- 驗證每個 subgraph 的 op coverage
- 檢查 subgraph 是否為 connected sub-DAG
- 檢查 output shape compatibility
- 檢查 retain legality
- 檢查 traversal order legality
- 檢查 tensor availability
- 檢查 fast memory capacity / OOM
- 依 tile 與 split-k step 重新計算 latency
- 用 `max(compute, memory)` 累加每個 step 的成本

### 現在版和官方的關係

目前的 evaluator 是：

- **依官方 `PROBLEM.md` 重建的 scorer**

不是：

- 官方原始 scorer source code 的直接移植

原因是目前官方 repo 只提供了文字規格與 examples，沒有附 evaluator 的實作碼。

### 目前最精確的說法

- 舊版 latency：solver 自己估，evaluator 幾乎直接採信
- 現在版 latency：evaluator 會依官方文字規格獨立重算
- 官方定義：以 `PROBLEM.md` 為準，是 tile-level / split-k-aware 的 roofline latency model

## 目前應如何解讀結果

- 目前 benchmark 表上的數字，應以 **新 evaluator 重算後的 latency** 為準
- 目前 `optimus` 是可用的，但還不是 paper 等級的完整實作
- 目前 `optimus` 的改善是真改善，但仍受限於搜尋空間、retain 策略與 cost model 的近似

## Backend 現況

- `baseline` 仍然保留，沒有被刪掉
- 預設 backend 是 `optimus`
- 可用環境變數切回 baseline：
  - `MLSYS_SOLVER=baseline ./build/mlsys <input.json> <output.json>`
- `conv_accelerator` module 已存在並已編進 build，但目前還沒有接進 solver 主流程

## 本輪進度：先對齊 scorer，再壓 benchmark

### 本輪保留下來的改動

這一輪的主軸不是接 `conv_accelerator`，而是先讓目前 contest solver 更直接對齊 scorer。

已完成的內容：

- 在 `source/evaluator.h/.cpp` 新增單一 subgraph 的公開評估介面
  - `EvaluateSubgraph(...)`
- 讓 `optimus` 的 group scoring / transition cost 直接使用 evaluator 的 subgraph simulation core
- 加入 retain-aware DP state 與 reconstruction
- 加入 scorer result cache，避免同一組 candidate 被重複完整模擬
- 加入 granularity prefilter
  - 先用 cheap proxy 排序
  - 再只把少數候選送進 evaluator

### 這一輪觀察到的效果

目前保留下來的版本，其 benchmark 結果如下：

- `mlsys-2026-1`: `405875`
- `mlsys-2026-5`: `915215`
- `mlsys-2026-9`: `1.64506e+08`
- `mlsys-2026-13`: `1.66406e+08`
- `mlsys-2026-17`: `5.01369e+06`

和上一輪相比，最明顯的改善是：

- `mlsys-2026-5`: `978671 -> 915215`

這代表 solver 直接使用 scorer 的 simulation core 之後，至少在部分 benchmark 上已經能做出更好的 partition / granularity 決策。

### 這一輪試過但已回退的內容

這一輪也試過更激進地放寬 group-size heuristic，希望直接吃掉更多 subgraph boundary。

結果是：

- `mlsys-2026-17` 沒有變好
- runtime 卻大幅上升，約到 80 秒等級

因此這個方向已經先回退，沒有保留在目前版本。

### 目前版本的取捨

目前保留的是一個比較穩定的版本：

- scorer 對齊程度比之前高
- 至少在 `mlsys-2026-5` 有明顯收益
- runtime 雖然增加，但還在可接受範圍
- 尚未為了追更大的搜尋空間而讓整體 runtime 完全失控

### 下一步

下一步不應再單純暴力放大 group size，而應該：

- 改 candidate generation 本身
- 減少 topo-interval partition 帶來的限制
- 讓 solver 看得到更有價值的合法 group

這樣才比較有機會在不把 runtime 撐爆的前提下，繼續把 `mlsys-2026-9 / 13 / 17` 往下壓。

## 目前狀況摘要

### 目前有哪些 backend / mode

- `baseline` 還保留著，沒有被刪掉
- `optimus` 是目前預設 backend
- `conv_accelerator` module 已存在，也已編進 build
- 但 `conv_accelerator` 還沒有接進 solver 主流程

目前切換方式：

- baseline:
  - `MLSYS_SOLVER=baseline ./build/mlsys <input.json> <output.json>`
- optimus:
  - 直接執行 `./build/mlsys ...`

此外，`optimus` 內部還有兩種 candidate generation mode：

- `interval`
  - 原本的方法
  - 目前仍是預設
- `seed-growth`
  - 新增的方法
  - 可用 `MLSYS_OPTIMUS_CANDIDATES=seed` 切換

### 目前 latency 的可信度

目前 `source/evaluator.cpp` 已經是獨立 scorer，而不是單純把 solver 填的 latency 加總。

它現在會：

- 檢查 subgraph legality
- 檢查 retain / traversal / tensor availability
- 檢查 OOM / fast memory capacity
- 重新計算 tile-level / split-k-aware latency

因此目前 benchmark 表上的數字，應以 evaluator 重算後的 latency 為準。

### 目前 `optimus` 做到哪

目前 `optimus` 已經有兩條 candidate generation 路徑：

1. `interval`
- 以 topo-order 連續區間產生 candidate
- search space 較窄，但目前較穩

2. `seed-growth`
- 先從 seed op 沿 graph 擴張，長出 candidate
- 再交給原本的 DP 去選，而不是直接貪婪出整個 schedule

目前 seed-growth 已經不是壞掉的狀態，至少在 benchmark 1 和 5 上已能追平 interval。

### 目前已知結果

目前穩定版本下，`optimus` 的 benchmark 結果大致是：

- `mlsys-2026-1`: `405875`
- `mlsys-2026-5`: `915215`
- `mlsys-2026-9`: `1.64506e+08`
- `mlsys-2026-13`: `1.66406e+08`
- `mlsys-2026-17`: `5.01369e+06`

seed-growth 目前已驗證：

- `benchmark 1`
  - `interval`: `405875`
  - `seed`: `405875`
- `benchmark 5`
  - `interval`: `915215`
  - `seed`: `915215`

## `conv_accelerator` 已接進 active path

### 目前接法

`conv_accelerator` 不再只是 groundwork，現在已新增一條獨立 backend：

- `optimus_conv`
  - 切換方式：
    - `MLSYS_SOLVER=optimus_conv ./build/mlsys <input.json> <output.json>`

這條路徑會：

- 保留官方 `evaluator` 當最終 legality / latency scorer
- 保留原本 `optimus` 的 DP / partition 架構
- 讓 `conv_accelerator` 參與 candidate / granularity 的 ranking guidance

也就是說，現在的接法是：

- `contest evaluator` 負責最終打分
- `conv_accelerator` 負責提供更 paper-like 的 chain memory heuristic

### 目前不是什麼

這一版還不是完整 paper backend，因為目前題目的 `Problem` schema 仍然沒有：

- `kernel size`
- `padding`
- `channel schema`
- 真正的 conv weights / feature-map layout

所以現在 `optimus_conv` 用的是 **pseudo-conv mapping**：

- 將 tensor `height` 視為 spatial height
- 將 tensor `width` 視為 channel-like dimension
- `MatMul` 的 reduction `K` 映射成 pseudo conv 的 input channels
- `Pointwise` 以 `1x1` conv-like op 近似
- terminal tile 以目前的 `[w, h, k]` 映射成：
  - `output_height <- h`
  - `output_channels <- w`

這樣做的目的不是取代官方模型，而是把 `conv_accelerator` 的：

- on-chip working set 分析
- parameter refill 分析
- ISOA / subgroup heuristic
- memory traffic estimate

接到現在的 search path 裡面。

### 目前效果

已確認：

- `optimus_conv` backend 可以正常編譯並執行
- `conv_accelerator` 已經是 active path，而不是未使用模組

目前觀察到的 benchmark 結果：

- `mlsys-2026-5`
  - `optimus`: `915215`
  - `optimus_conv`: `938153`

因此目前的結論是：

- `conv_accelerator` 已經接進去了
- 但這一版比較像 **paper-like guidance backend**
- 還沒有在 released benchmark 上穩定贏過目前的 `optimus`

### 新增的切換與參數

- backend:
  - `MLSYS_SOLVER=optimus_conv`
- backend:
  - `MLSYS_SOLVER=optimus_conv_v2`
- conv dataflow:
  - `MLSYS_OPTIMUS_CONV_DATAFLOW=os|ws|is|rs`
  - 預設：`rs`

## `optimus_conv_v2`

### 修改方向

為了避免舊版 `optimus_conv` 直接把 pseudo-conv penalty 加到 contest proxy latency 上，新增了 `optimus_conv_v2`。

這一版的重點是：

- 不再做全域 additive penalty
- 先用原本 contest proxy 排序
- 只對 top-K 候選做 conv-guided local rerank
- 目前只使用 normalized traffic ratio
- `parameter_refills` 與 `subgroup_count` 目前只用於輕量 pruning

### 目前規則

- 只對下列 candidate 啟用 conv rerank：
  - `ops.size() >= 2`
  - 含至少一個 `MatMul`
  - 是 linear chain
- rerank 只看 proxy top `8`
- normalized penalty 目前為：
  - `proxy_score * (1 + 0.15 * conv_traffic_time / proxy_score)`
- 輕量 prune：
  - `conv working set > fast_memory_capacity`
  - `parameter_refills > 2 * ops.size()`
  - `subgroup_count > 4`

### 目前結果

- `mlsys-2026-5`
  - `optimus`: `915215`
  - `optimus_conv`: `938153`
  - `optimus_conv_v2`: `915215`
- `mlsys-2026-1`
  - `optimus`: `405875`
  - `optimus_conv_v2`: `405875`

目前結論：

- `optimus_conv_v2` 已經比舊版 `optimus_conv` 穩定
- 至少已能追平目前的 `optimus`
- 下一步才是看能不能在 `9 / 13 / 17` 上找出真正的收益，而不是先讓 conv guidance 破壞原本排序

## 四個 Backend Benchmark 比較

### 差異公式

各 backend 相對於 `baseline` 的差異百分比使用下列公式：

`(backend_latency - baseline_latency) / baseline_latency * 100%`

- 負值：代表比 `baseline` 更好
- 正值：代表比 `baseline` 更差

### 比較表

| Benchmark | baseline | optimus | optimus Δ% | optimus_conv | optimus_conv Δ% | optimus_conv_v2 | optimus_conv_v2 Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlsys-2026-1` | `471501` | `405875` | `-13.92%` | `405875` | `-13.92%` | `405875` | `-13.92%` |
| `mlsys-2026-5` | `1.01362e+06` | `1.01391e+06` | `+0.03%` | `1.03685e+06` | `+2.29%` | `1.01391e+06` | `+0.03%` |
| `mlsys-2026-9` | `1.67531e+08` | `1.64506e+08` | `-1.81%` | `1.64506e+08` | `-1.81%` | `1.64506e+08` | `-1.81%` |
| `mlsys-2026-13` | `1.66415e+08` | `1.66406e+08` | `-0.01%` | `1.66406e+08` | `-0.01%` | `1.66406e+08` | `-0.01%` |
| `mlsys-2026-17` | `5.08185e+06` | `5.01369e+06` | `-1.34%` | `5.01369e+06` | `-1.34%` | `5.01369e+06` | `-1.34%` |

### 結論

- `optimus` 目前仍然是主版本，整體來看最穩定。
- `optimus_conv` 第一版會在部分 benchmark 把排序弄壞，`mlsys-2026-5` 最明顯，退化到 `+2.29%`。
- `optimus_conv_v2` 已經把 `optimus_conv` 拉回來，目前大多數 benchmark 都能追平 `optimus`。
- `mlsys-2026-1` 是目前 `optimus` family 收益最明顯的 case，約 `-13.92%`。
- `mlsys-2026-9` 與 `mlsys-2026-17` 有小幅改善，約 `-1.81%` 與 `-1.34%`。
- `mlsys-2026-13` 幾乎沒有改善，代表目前 search space 與 cost model 仍然不夠強。

也就是說：

- 新方法已經接進來了
- 舊方法沒有被蓋掉
- 但新方法目前還沒有穩定超過舊方法

### 目前真正卡住的地方

目前主要卡在 search quality 與 runtime，不是 correctness。

具體來說：

- `interval` 方法較穩，但 search space 太窄
- `seed-growth` 方法已經能用，但在大圖上還沒有穩定超車
- 因為 solver 現在更直接對齊 scorer，所以每次 search 的成本也更高
- 一旦把搜尋空間放太大，runtime 很容易迅速上升
- `conv_accelerator` 雖然已經存在，但目前對 contest benchmark 還沒有直接加成

### 目前最精確的一句話

目前 repo 的狀態是：

- scorer 已經站穩
- backend 已經分層
- `optimus` 已經有新舊兩種 candidate 方法可切換
- 但接下來真正要繼續打的是 search quality，不是 latency correctness
