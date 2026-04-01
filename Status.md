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

