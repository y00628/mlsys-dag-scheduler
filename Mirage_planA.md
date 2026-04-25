# Mirage Part A Implementation Plan

This plan breaks down Mirage Part A (IR / Graph Representation) into 3 main stages, each with concrete tasks and suggested function/class designs. Each step is independently verifiable and extensible.

---

## Stage 1: Core Data Structures & Multi-Level Graph Hierarchy

### 1.1 Tensor Class Design
- [ ] Define a `Tensor` class with attributes:
  - `shape` (multi-dimensional)
  - `dtype` (data type)
  - `layout` (memory layout)
  - `memory_type` (Global/Shared/Register)
  - `swizzle` (for STensor only, to avoid bank conflict)

### 1.2 KNGraph (Kernel Graph)
- [ ] Define `KNNode` (kernel-level operator)
- [ ] Define `KNEdge` (DTensor, device memory tensor)
- [ ] Define `KNGraph` structure, including:
  - `nodes: List[KNNode]`
  - `edges: List[KNEdge]`
  - Methods: `add_node()`, `add_edge()`, `get_consumers()`, `get_producers()`, etc.

### 1.3 TBGraph (Threadblock Graph)
- [ ] Define `TBNode` (block-level operator)
- [ ] Define `TBEdge` (STensor, shared memory tensor)
- [ ] Define `TBGraph`, inheriting from `KNGraph`, and add:
  - `grid_dim` (block configuration)
  - `for_loop_dim` (reduction dimension)
  - Methods: `add_tbnode()`, `add_tbedge()`

### 1.4 Thread Graph (Optional)
- [ ] Define thread-level structure (if finer-grained optimization is needed)

---

## Stage 2: Tensor Metadata, Indexing Mappings & Operator System

### 2.1 Tensor Attributes & Metadata
- [ ] `DTensor`/`STensor` classes inherit from `Tensor`, each with specific metadata
- [ ] `calc_tensor_strides()`: compute stride for each dimension
- [ ] `find_innermost_dim()`: analyze for coalesced access

### 2.2 Indexing Mappings
- [ ] `imap` (Input Map): defines how blocks partition input
- [ ] `omap` (Output Map): defines how block outputs are stitched together
- [ ] `fmap` (For-loop Map): defines tensor partitioning within for-loops
- [ ] Each mapping should have `parse()` and `apply(block_id, loop_idx)` functions

### 2.3 Operator Type System
- [ ] Define Operator Type Enums (`KNOperatorType`, `TBOperatorType`)
- [ ] Supported types:
  - Load/Store: InputIterator, OutputSaver
  - Core compute: Matmul, Conv, EwAdd, EwMul, EwExp, Accum
  - Auxiliary/Transform: Reshape, Sqrt, SiLU, Repeat
  - Custom: GraphDefinedOperator (can point to a lower-level TBGraph)

---

## Stage 3: Validity Checking, Recursive Expansion & Testing

### 3.1 Graph Validity Checking
- [ ] `check_validity()` function to verify:
  - All operators' input/output tensor shapes and attributes
  - KNGraph tensors fit in device memory
  - TBGraph tensors fit in shared memory
  - Thread Graph tensors fit in register file
  - For-loop body structure constraint (InputIterator → Accumulator → OutputSaver)

### 3.2 GraphDefinedOperator Support
- [ ] `GraphDefinedOperator` class, allowing a KNNode to encapsulate a full TBGraph
- [ ] Support recursive expansion (KNGraph → TBGraph → Thread Graph)

### 3.3 Recursive Expansion & Codegen Preparation
- [ ] `expand_graph()` recursive function to unfold KNGraph into a hierarchical structure
- [ ] Mark which parts require Triton/CUDA PTX codegen (for future codegen stages)

### 3.4 Unit Testing & Validation
- [ ] Write unit tests for each class/function
- [ ] Construct simple KNGraph/TBGraph/Thread Graph examples to validate legality checks and expansion

---

# Appendix: Suggested Function & Class Interfaces

```cpp
// Tensor.h
class Tensor {
  std::vector<int> shape;
  std::string dtype;
  std::string layout;
  enum MemoryType { Global, Shared, Register } memory_type;
  // ... swizzle, stride, etc.
};

// Graph.h
class KNNode { /* ... */ };
class KNEdge { /* ... */ };
class KNGraph {
  std::vector<KNNode> nodes;
  std::vector<KNEdge> edges;
  void add_node(const KNNode&);
  void add_edge(const KNEdge&);
  // ...
};

class TBNode { /* ... */ };
class TBEdge { /* ... */ };
class TBGraph : public KNGraph {
  std::vector<TBNode> tb_nodes;
  std::vector<TBEdge> tb_edges;
  std::vector<int> grid_dim;
  std::vector<int> for_loop_dim;
  // ...
};

// OperatorType.h
enum class KNOperatorType { Matmul, Conv, EwAdd, ... };
enum class TBOperatorType { InputIterator, OutputSaver, ... };

// Validity.h
bool check_validity(const KNGraph&);
```

---

If you need further breakdowns, class implementations, or test cases for any stage, just specify the stage or task!