
# UrbanFloodBench GNN 训练流水线
<!-- 此文档记录代码模块间的完整关联逻辑、环境配置与使用方法 -->


这是一个基于 **几何深度学习 (Geometric Deep Learning)** 的模块化流水线，专为时空城市洪水预测设计。它使用异构图神经网络 (HeteroGNN) 来模拟地下管网与地表漫流的耦合系统。

## 🎯 问题概述 

- **1D 节点 (Manholes)**: 地下排水管网。
- **2D 节点 (Cells)**: 地表地形网格。水流在此汇聚，通过耦合连接流入地下。
- **核心挑战**: 模型需要仅通过图拓扑结构（`edge_index`）和时序动态，**隐式学习**出哪些节点是排水口（Outlets）。

## 🏗️ 架构设计

```mermaid
graph LR
    Input[静态 + 动态特征] --> Encoder[类型专用 MLP]
    Encoder --> Processor[循环 GNN (GRU + HeteroConv)]
    Processor --> Decoder[类型专用 MLP]
    Decoder --> Output[下一时刻水位预测]

```

### 关键组件

1. **Encoder**: 针对 1D 和 2D 节点分别使用 MLP 将原始特征映射到隐层维度 `D=128`。
2. **Processor**: GRU 风格的循环单元，将其内部的线性变换替换为 `HeteroConv`。
* 使用 **GATv2Conv** (图注意力机制) 处理同构边（1D-1D, 2D-2D）。
* **关键修正**: 对于异构耦合边（1D-2D），显式禁用了自环 (`add_self_loops=False`) 以避免维度错误。


3. **Decoder**: MLP 将更新后的隐状态映射回标量（水位）。

## 📁 代码结构与模块关联

### 文件一览

| 文件 | 职责 | 核心类 / 函数 |
|---|---|---|
| `config.py` | 所有超参数的唯一来源 | `ModelConfig`, `TrainingConfig` |
| `dataset.py` | 从 CSV 构建 `HeteroData` 图；按需加载动态事件 | `UrbanFloodDataset`, `load_event()` |
| `model.py` | 异构 GNN 模型定义（Encoder → GRU-GNN → Decoder）| `HeteroFloodGNN` |
| `train.py` | 完整训练循环（Teacher Forcing、Truncated BPTT、物理损失）| `train()`, `physics_mass_conservation_loss()` |
| `inference.py` | 自回归推理，生成 Kaggle 提交 CSV | `run_inference_stream()` |
| `make_submission.py` | 合并多模型预测 CSV → 最终提交文件 | `merge_inputs()`, `align_to_template()` |
| `preprocess.py` | 将 CSV 批量转换为 `.pt` 缓存（加速 `load_event`）| `process_split()` |
| `test_pipeline.py` | 端到端单元测试，验证数据加载和模型前向传播 | — |

---

### 模块依赖关系（import 层级）

```
config.py           ← 无外部依赖（纯 dataclass）
    ↑
    ├── model.py    ← 读取 ModelConfig；依赖 torch_geometric
    │       ↑
    ├── dataset.py  ← 独立；依赖 torch_geometric.data
    │       ↑
    ├── train.py    ← 导入 config / dataset / model；是训练入口
    ├── inference.py← 导入 config / dataset / model；读 checkpoint 写 CSV
    └── test_pipeline.py ← 导入 config / dataset / model；只读不写
         
make_submission.py  ← 独立脚本；仅依赖 pandas / numpy（无 PyTorch）
preprocess.py       ← 独立脚本；仅依赖 pandas / numpy / torch（无 PyG）
```

**关键原则**：`config.py` 是所有模块的起点，修改超参数只需改这一个文件。

---

### 数据流（端到端）

```
原始 CSV 文件
  Models/Model_{id}/{split}/
    ├── *_static.csv              → dataset.py::process()
    ├── *_edge_index.csv          → 构建 HeteroData 图结构 (.pt 缓存在 processed/)
    └── event_*/
        ├── *_dynamic_all.csv     → dataset.py::load_event()
        └── event_data.pt         ← preprocess.py 预生成（可选，加速 10×）

HeteroData（图结构）
  + event_data（动态时序，[T, N, D]）
      ↓
  model.py::HeteroFloodGNN.forward()   ← 每个时间步调用一次
      ↓
  pred_dict {'manhole': [N1,D1], 'cell': [N2,D2]}
      ↓
  train.py → 计算损失、反传、更新权重 → checkpoints/best_model.pt
      or
  inference.py → 写入 submission_{model_id}.csv
      ↓
  make_submission.py → final_submission_filled.csv（对齐 sample_submission.csv）
```

---

### 各模块详细说明

#### `config.py` — 超参数中心

定义两个 `dataclass`，无副作用：

- **`ModelConfig`**：网络结构参数（`hidden_dim=128`、`num_gnn_layers=3`、`use_gatv2=True` 等）
- **`TrainingConfig`**：训练参数 + 事件分配逻辑
  - `model_id` 触发 `__post_init__` 自动按 8:2 划分 train/val 事件列表
  - Model_1 共 68 事件；Model_2 共 69 事件

#### `dataset.py` — 图数据加载

`UrbanFloodDataset(root, model_id, split)` 继承 `InMemoryDataset`：

1. **`process()`**：首次运行时将静态 CSV → `HeteroData` 图，缓存至 `processed/model_{id}_{split}_graph.pt`
2. **`load_event(event_folder)`**：返回字典 `{'manhole': [T,N1,2], 'cell': [T,N2,3], '1d_edges': [T,E1,2], '2d_edges': [T,E2,2], 'timesteps': T}`
   - 优先读取 `event_data.pt`（快速模式）；缺失时回退到 CSV 解析（慢速模式）

节点特征维度：
- Manhole 静态 4 维：`depth, invert_elevation, surface_elevation, base_area`
- Cell 静态 6 维：`area, roughness, min_elevation, elevation, aspect, curvature`
- Manhole 动态 2 维：`water_level, inlet_flow`
- Cell 动态 3 维：`rainfall, water_level, water_volume`

#### `model.py` — HeteroFloodGNN

`HeteroFloodGNN(config, manhole_static_dim=4, cell_static_dim=6, manhole_dynamic_dim=2, cell_dynamic_dim=3)`

三阶段前向传播（每步调用一次）：

1. **Encoder**：`静态 + 归一化动态` → MLP → 隐向量 `[N, D=128]`
2. **Processor**（循环 `num_recurrent_steps` 次）：GRU-GNN，三组 `HeteroConv` 实现 reset / update / candidate 门
3. **Decoder（残差）**：预测归一化增量 `Δx_norm`，叠加当前输入还原真实值

返回 3-tuple：`(pred_dict, h_dict, cell_to_cell_flow)`

#### `train.py` — 训练主循环

- `compute_stats_from_events()` → 从训练事件计算 mean/std，注入模型 buffer
- `StandardizedRMSELoss` → Cell 忽略 index 0（rainfall），仅对 `water_level` 和 `water_volume` 求损失
- `physics_mass_conservation_loss()` → 软约束 L1 损失，权重 `λ=0.1`
  - **内部使用 PyTorch 原生 `scatter_add_`（无需 `torch_scatter` 包）**
- `train_one_event()` → Truncated BPTT，`chunk_size=20`，每 chunk 独立 backward

#### `inference.py` — 推理与提交

- 加载 `checkpoints/best_model.pt`
- 对 `Models/Model_{id}/test/` 下所有事件自回归预测
- 流式写 CSV（逐 event 写盘，内存安全）
- 生成 `submission_{model_id}.csv`

#### `make_submission.py` — 合并提交

将多个模型的预测 CSV 合并，并按照 `sample_submission.csv` 的 `row_id` 对齐，输出 `final_submission_filled.csv`。

#### `preprocess.py` — 可选预处理

将所有事件的 CSV 批量转换为 `.pt` 张量文件，使 `load_event()` 速度提升约 10 倍：

```bash
python preprocess.py --model_id 1 --split train
python preprocess.py --model_id 1 --split test
python preprocess.py --model_id 2 --split train
python preprocess.py --model_id 2 --split test
```

## 🚀 快速开始

### 1. 环境准备（RTX 5060 / Blackwell · CUDA 13.0）

> **注意**：`torch_scatter` 不再是必需依赖。`train.py` 已使用 PyTorch 原生 `scatter_add_` 替换，可直接跳过该包的安装。

```bash
# 步骤 1：创建并激活 conda 环境
conda create -n floodenv python=3.10 -y
conda activate floodenv

# 步骤 2：安装 PyTorch 2.9.1（CUDA 13.0，适配 RTX 5060 Blackwell 架构）
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130

# 步骤 3：安装 PyTorch Geometric
pip install torch_geometric

# 步骤 4：其他依赖
pip install pandas numpy tqdm
```

验证安装：

```bash
conda run -n floodenv python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
import torch_geometric
print('PyG:', torch_geometric.__version__)
"
```

> 如果需要从 CPU 环境运行（无 GPU），将 `torch==2.9.1+cu130` 替换为 `torch` 即可。

### 2. 数据目录结构

请确保数据放置在以下路径 (无需 Shapefiles):

```
Models/Model_2/train/
  ├── 1d_nodes_static.csv       # 井盖静态特征
  ├── 2d_nodes_static.csv       # 网格静态特征
  ├── 1d_edge_index.csv         # 1D 拓扑
  ├── 2d_edge_index.csv         # 2D 拓扑
  ├── 1d2d_connections.csv      # 1D-2D 耦合
  └── event_1/                  # 动态事件文件夹
      ├── 1d_nodes_dynamic_all.csv
      ├── 2d_nodes_dynamic_all.csv
      └── timesteps.csv

```

### 3. 开始训练

1. 默认情况下 `config.py` 会根据 `model_id` 自动从对应的 `Models/Model_{id}/train/` 目录中选择 **所有事件** 并以 **8:2（train:val）** 划分：
   - `model_id=1` → 使用 Model_1 的事件（全部分配，前 80% 用于训练，后 20% 用于验证）
   - `model_id=2` → 使用 Model_2 的事件（同上）

   如果你想手动指定事件集合，可以在 `TrainingConfig` 中传入 `train_events` / `val_events`：

```python
from config import TrainingConfig
train_cfg = TrainingConfig(model_id=2)
# 或者覆盖为自定义列表
train_cfg.train_events = ['event_1', 'event_2', ...]
train_cfg.val_events = ['event_80','event_81']
```

2. 运行训练脚本：

```bash
python train.py
```

脚本会自动执行以下步骤：

* 加载并处理静态图结构 (保存为 `.pt` 文件)。
* 计算训练集的均值/标准差（动态特征归一化）并注入到训练流程中。注意：`train.py` 已修复 `compute_stats_from_events`，它现在同时兼容 `numpy.ndarray` 和 `torch.Tensor`，不会因数据类型不同而报错。
* 开始自回归训练 (带 Teacher Forcing 衰减)。
* 在验证集上计算验证损失并保存 `checkpoints/best_model.pt`（仅当验证损失比历史最好值更优时覆盖）。

### 4. 验证与测试

使用提供的测试脚本检查流水线各环节是否正常：

```bash
python test_pipeline.py

```

## 📊 数据处理细节

### 图构建 (`dataset.py`)

`UrbanFloodDataset` 构建了一个 `HeteroData` 对象：

* **节点类型**: `manhole` (198个), `cell` (4299个)
* **边类型**:
* `(manhole, to_manhole, manhole)`: 1D 管网流
* `(cell, to_cell, cell)`: 2D 地表流
* `(manhole, to_cell, cell)`: 溢流 (Overflow)
* `(cell, to_manhole, manhole)`: 排水 (Drainage)



**注意**: 数据加载使用了 `weights_only=False` 以支持加载复杂的 PyG 对象。

### 动态数据加载

为了节省内存，动态特征（如降雨、水位）是按需加载的：

```python
# 正确的加载方式 (类方法)
dataset = UrbanFloodDataset(root="./", model_id=2)
event_data = dataset.load_event('event_1') 
# 返回: {'manhole': [T, N1, 2], 'cell': [T, N2, 3]}

```

## 🎓 核心训练策略

### 1. Teacher Forcing (教学强制)

为了解决长序列预测的误差累积问题：

* **训练初期**: TF Ratio = 1.0 (完全使用真实值作为下一步输入)。
* **训练后期**: TF Ratio 线性衰减至 0.2 (主要使用模型自己的预测值)。
* **梯度截断**: 在每一步反向传播后，使用 `.detach()` 切断隐状态的梯度流，防止 `RuntimeError: Trying to backward through the graph a second time`。

### 2. 标准化 RMSE Loss

根据比赛要求，不同节点类型的 Loss 权重相等：

```python
Loss = 0.5 * (RMSE_manhole / std_manhole) + 0.5 * (RMSE_cell / std_cell)

```

---

### Model Architecture Upgrades (DUALFloodGNN Inspired)

本节记录了受 **DUALFloodGNN** 论文启发、对模型架构与训练流程所做的四项核心升级。每项改进都直接针对自回归洪水预测任务中已知的痛点，并通过代码层面的具体实现加以落地。

---

#### 1. 残差预测 (Residual Prediction)

**问题**：原始模型的 `Decoder` 直接输出 $\hat{x}_{t+1}$ 的绝对值。在自回归推理时，每一步的绝对值误差会按序列长度线性叠加，导致长时预测时水位持续偏移（误差漂移）。

**改进**：解码器改为预测归一化空间中的**增量** $\Delta x_\text{norm}$，再叠加当前时刻的输入作为残差跳跃连接：

$$\hat{x}_{t+1,\text{norm}} = x_{t,\text{norm}} + \Delta x_\text{norm}$$

$$\hat{x}_{t+1,\text{real}} = \hat{x}_{t+1,\text{norm}} \times \sigma + \mu$$

**优势**：
- 学习目标从 $[0, \infty)$ 域的绝对水位值收窄为均值接近 $0$ 的小增量，梯度信号更干净、优化更稳定。
- 自回归误差以增量形式叠加，而非绝对值替换，累积速度显著减慢。
- 对应 `model.py` 中 `delta_manhole_norm`、`delta_cell_norm` 变量，形状 `[N, D]`。

---

#### 2. 隐式边流量预测 (Latent Edge Flow Prediction)

**问题**：原始模型仅预测节点状态（水位、水量），完全忽略了节点间的**水流传输过程**，无法感知水动力学中的流量-水位耦合关系。

**改进**：在 `__init__` 中新增 `self.edge_decoder`，以边两端节点的隐状态拼接作为输入，预测该边上的标量流量值 $Q$：

```
h_src || h_dst → [E, 2D] → edge_decoder → Q̂  [E, 1]
```

**实现细节**：
- 输入：`cat([h_src, h_dst])`，形状 `[E, 2*D]`，编码流量方向性（src→dst）。
- 数据对齐：`*_edges_dynamic_all.csv` 中的 `flow` 列可直接作为监督标签。
- 训练阶段可引入辅助监督损失；推理阶段作为隐变量供物理后处理使用。
- `forward` 方法返回签名从 2-tuple 升级为 **3-tuple**：`(pred_dict, h_dict, cell_to_cell_flow)`。

---

#### 3. 局部质量守恒物理损失 (Physics-informed Local Mass Conservation Loss)

**问题**：纯数据驱动的损失函数（RMSE）不对物理约束施加任何硬性限制，训练后的模型可能预测出在物理上"平白凭空生水"或"水量瞬间消失"的情况，违反浅水方程的连续性方程。

**改进**：在 `train.py` 中新增 `physics_mass_conservation_loss`，基于**浅水方程离散连续性方程**构建软约束损失：

$$\Delta V_i = \left(\sum_{j \to i} Q_{ji} - \sum_{i \to k} Q_{ik}\right) \cdot \Delta t + R_i$$

即：**ΔV = 流入 − 流出 + 降雨补给**

**关键设计决策**：

| 设计项 | 选择 | 理由 |
|---|---|---|
| 流量符号处理 | `flow_pos = ReLU(edge_flow)` | 有向边流量带符号；ReLU 分离正向流，避免反向流混入守恒方程导致符号对消 |
| 节点聚合 | `torch_scatter.scatter(..., reduce='sum')` | GPU 上高效的按索引稀疏加法，正确处理变长入度/出度 |
| 损失函数 | L1（绝对误差） | 比 MSE 更鲁棒，防止洪峰时期极大流量值主导梯度 |
| 损失权重 λ | 默认 `0.1` | 软约束，不主导 RMSE，但持续向守恒方向施加梯度压力 |

**总损失**：`loss = rmse_loss + 0.1 × physics_loss`

---

#### 4. 多步截断时序反向传播 (Truncated BPTT / Multi-step Rollout Loss)

**问题**：原始逐步反传（每步 `backward()` + `detach()`）等价于将模型退化为单步前馈网络——梯度永远无法跨越当前时刻，模型无法学习任何多步时间依赖。若改用完整 BPTT（全序列展开），T≈400–600 步的 GNN 计算图会立即触发显存 OOM。

**改进**：实现 **Truncated BPTT**，将长序列分割为长度 `chunk_size=20` 的子段交替执行前向-反向传播：

```
序列: ──[t=0..19]──|──[t=20..39]──|──[t=40..59]──| ...
                   ↑              ↑              ↑
              backward()     backward()     backward()
              step()         step()         step()
              h.detach()     h.detach()     h.detach()
```

**每个 chunk 边界的操作序列**（代码注释中标记 ①–⑦）：

1. **归一化**：`accumulated_loss / actual_chunk_steps`，保证尾 chunk 与满 chunk 梯度量级一致。
2. **反传**：`chunk_loss_normed.backward()`，梯度在 chunk 内完整展开 20 步。
3. **梯度裁剪**：`clip_grad_norm_`，防止多步依赖诱发梯度爆炸。
4. **更新**：`optimizer.step()`，每个 chunk 后立即更新参数。
5. **清零**：`optimizer.zero_grad()`，为下一 chunk 准备干净的梯度缓冲。
6. **释放计算图**：`accumulated_loss = None`，立即回收显存。
7. **截断隐状态**：`h_dict = {k: v.detach() ...}`，数值传递（序列状态保留），梯度截断（跨 chunk 反传阻断）。

**效果对比**：

| 策略 | 梯度跨步范围 | 显存占用 | 权重更新频率 |
|---|---|---|---|
| 逐步反传（原始） | 1 步 | O(1) | 每步 1 次 |
| 完整 BPTT | T 步（≈500） | O(T) → OOM | 每 event 1 次 |
| **Truncated BPTT（当前）** | **chunk_size=20 步** | **O(chunk)** | **每 chunk 1 次** |

---

## 🔧 常见故障排除

### Q: `OSError: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs` (torch_scatter)

**原因**：`torch_scatter` 包是针对特定 PyTorch 版本预编译的，当 PyTorch 版本更新后 `.so` 的符号名称发生变化，导致 `dlopen` 失败。

**根本修复（已应用）**：
- `train.py` 中已将 `from torch_scatter import scatter` **完全移除**，使用 PyTorch 原生 `tensor.scatter_add_()` 等价替换。
- 代码行为完全一致，无任何功能退化。
- 如果遇到此错误，只需确保使用最新的 `train.py`，无需手动安装 `torch_scatter`。

**同时需要**：确保 PyTorch 版本 ≥ 2.0（否则 `torch_geometric` 2.7.0 会因 `torch.jit.script` API 变化而崩溃）：
```bash
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

---

### Q: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**A:** 这是因为 GRU 的输入不仅包含当前特征 ，还包含上一时刻的隐状态 。

* **修复**: 确保卷积层初始化时 `in_channels = 2 * hidden_dim`。

### Q: `ValueError: add_self_loops attribute set to True`

**A:** GATv2Conv 默认会加自环，但这在异构边（如 1D->2D）上是非法的。

* **修复**: 在 `model.py` 中，针对 `(manhole, to_cell, cell)` 等异构边设置 `add_self_loops=False`。

### Q: `AttributeError: 'HeteroData' has no attribute 'load_event'`

**A:** 我们优化了代码结构，将加载逻辑移回了 Dataset 类。

* **修复**: 使用 `dataset.load_event(...)` 而不是 `data.load_event(...)`。

## 📝 引用

```bibtex
@misc{urbanfloodbench2026,
  title={UrbanFloodBench GNN Pipeline (Model 2)},
  author={Chishan},
  year={2026},
  framework={PyTorch Geometric}
}

```


# UrbanFloodBench GNN Training Pipeline

A robust, modular **Geometric Deep Learning** pipeline for spatio-temporal urban flood forecasting using Heterogeneous Graph Neural Networks (HeteroGNN).

## 🎯 Problem Overview

This pipeline models coupled 1D-2D urban flood systems:
- **1D Nodes (Manholes)**: Underground drainage network
- **2D Nodes (Cells)**: Surface terrain
- **Challenge**: Learn implicit boundary conditions (outlets) from topology and temporal dynamics

## 🏗️ Architecture

```
Input (Static + Dynamic) → Encoder (MLP) → Recurrent Processor (GRU-GNN) → Decoder (MLP) → Water Level Prediction
```

### Key Components

1. **Encoder**: Node-type-specific MLPs project features to hidden dimension `D=128`
2. **Processor**: GRU-style recurrent GNN where linear ops are replaced by `HeteroConv`
   - Uses **GATv2Conv** (Graph Attention) or **GENConv**
   - Implements reset/update/candidate gates with message passing
3. **Decoder**: MLPs project hidden states to scalar water level predictions

## 📁 Code Structure

```
.
├── dataset.py          # HeteroData construction from CSV files
├── model.py            # HeteroFloodGNN architecture
├── train.py            # Training loop with teacher forcing
├── inference.py        # Generate Kaggle submissions
├── config.py           # Hyperparameters
├── build_static.py     # Graph topology exploration utility
└── README_GNN.md       # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Required packages
pip install torch==2.9.1+cu130 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.9.1+cu130.html
pip install pandas numpy tqdm
```

### 2. Data Structure

Expected directory layout:
```
Models/Model_2/train/
  ├── 1d_nodes_static.csv       # Manhole static features
  ├── 2d_nodes_static.csv       # Cell static features
  ├── 1d_edge_index.csv         # Manhole-manhole connections
  ├── 2d_edge_index.csv         # Cell-cell connections
  ├── 1d2d_connections.csv      # Coupling between 1D and 2D
  └── event_1/
      ├── 1d_nodes_dynamic_all.csv
      ├── 2d_nodes_dynamic_all.csv
      └── timesteps.csv
```

### 3. Training

```python
# Example training script
from config import ModelConfig, TrainingConfig
from train import train

model_config = ModelConfig(
    hidden_dim=128,
    num_gnn_layers=3,
    num_recurrent_steps=3,
    use_gatv2=True,
    num_heads=4,
    dropout=0.1
)

train_config = TrainingConfig(
    model_id=2,
    learning_rate=1e-3,
    num_epochs=50,
    teacher_forcing_ratio_start=1.0,
    teacher_forcing_ratio_end=0.2,
    teacher_forcing_decay_epochs=30,
    device="cuda"
)

train(model_config, train_config)
```

Or via command line:
```bash
python train.py
```

### 4. Generate Submission

Use the trained checkpoint to run inference and produce per-model prediction CSVs (one file per model):

```bash
python inference.py --checkpoint ./checkpoints/best_model.pt --model_id 2 --output submission_2.csv
```

If you have multiple model outputs (e.g., `submission_1.csv`, `submission_2.csv`), the new helper `make_submission.py` can merge them into a Kaggle-ready submission aligned to the `sample_submission.csv` template.

```bash
# Example: merge prediction fragments and fill template
python make_submission.py
# Configure INPUT_FILES, SAMPLE_SUBMISSION_FILE, FINAL_OUTPUT_FILE at top of make_submission.py as needed
```

Notes:
- `make_submission.py` performs a two-stage merge: it creates a compact intermediate file and then streams a final aligned submission using `sample_submission.csv` as the canonical template.
- Missing values are filled with `0.0` and NaNs are corrected to `0.0` by default; summary stats are printed after completion.


## 📊 Data Processing

### Static Features

**Manholes (1D):**
- `depth`, `invert_elevation`, `surface_elevation`, `base_area`

**Cells (2D):**
- `area`, `roughness`, `min_elevation`, `elevation`, `aspect`, `curvature`

### Dynamic Features (per timestep)

**Manholes:**
- `water_level`, `inlet_flow`

**Cells:**
- `rainfall`, `water_level`, `water_volume`

### Graph Construction

The `UrbanFloodDataset` builds a `HeteroData` object with:
- **Node types**: `manhole`, `cell`
- **Edge types**:
  - `(manhole, to_manhole, manhole)`: 1D drainage network
  - `(cell, to_cell, cell)`: 2D surface flow
  - `(manhole, to_cell, cell)`: 1D→2D coupling
  - `(cell, to_manhole, manhole)`: 2D→1D coupling

## 🎓 Key Training Features

### 1. Teacher Forcing with Scheduled Decay

- **Initial**: 100% ground truth at timestep `t` to predict `t+1`
- **Decay**: Linear decay over 30 epochs to 20%
- **Final**: Mostly autoregressive (model uses own predictions)

```python
tf_ratio = teacher_forcing_ratio_start + progress * (end - start)
use_gt = np.random.rand() < tf_ratio
```

### 2. Standardized RMSE Loss

Per competition requirements:
```python
RMSE_manhole = sqrt(mean((pred - target)^2))
RMSE_cell = sqrt(mean((pred - target)^2))

Standardized_RMSE = (RMSE_manhole/std_manhole + RMSE_cell/std_cell) / 2
```

### 3. Autoregressive Validation

During validation/test: **NO teacher forcing**
- Use ground truth only for first timestep
- All subsequent predictions use model outputs

## 🔧 Hyperparameter Tuning

Key parameters to tune in `config.py`:

```python
# Architecture
hidden_dim: 64, 128, 256
num_recurrent_steps: 1, 3, 5
num_heads: 2, 4, 8  # For GATv2Conv

# Training
learning_rate: 1e-4, 1e-3, 5e-3
teacher_forcing_decay_epochs: 20, 30, 40
gradient_clip: 0.5, 1.0, 5.0
```

## 📈 Monitoring Training

The training loop logs:
```
Epoch 1: Train Loss=0.3421, Val Loss=0.4123, TF Ratio=1.00, LR=1.00e-03
Epoch 10: Train Loss=0.2156, Val Loss=0.2789, TF Ratio=0.73, LR=8.91e-04
Epoch 30: Train Loss=0.1345, Val Loss=0.1876, TF Ratio=0.20, LR=5.00e-04
```

Checkpoints saved to `./checkpoints/`:
- `best_model.pt`: Best validation loss
- `model_epoch{N}.pt`: Periodic saves

## 🧪 Testing & Debugging

### 1. Verify Dataset Loading

```python
from dataset import UrbanFloodDataset

dataset = UrbanFloodDataset(root="./", model_id=2)
data = dataset.get(0)

print(f"Manholes: {data['manhole'].x_static.shape}")  # [N1, 4]
print(f"Cells: {data['cell'].x_static.shape}")  # [N2, 6]
print(f"1D edges: {data['manhole', 'to_manhole', 'manhole'].edge_index.shape}")
print(f"2D edges: {data['cell', 'to_cell', 'cell'].edge_index.shape}")

# Load an event
event_data = data.load_event('event_1')
print(f"Manhole dynamics: {event_data['manhole'].shape}")  # [T, N1, 2]
print(f"Cell dynamics: {event_data['cell'].shape}")  # [T, N2, 3]
```

### 2. Test Model Forward Pass

```python
from model import HeteroFloodGNN
from config import ModelConfig
import torch

config = ModelConfig(hidden_dim=128)
model = HeteroFloodGNN(config, 4, 6, 2, 3)

# Simulate one timestep
manhole_dyn = torch.randn(121, 2)  # [N1, D1]
cell_dyn = torch.randn(5213, 3)  # [N2, D2]

pred_dict, h_dict = model(data, manhole_dyn, cell_dyn)
print(f"Manhole predictions: {pred_dict['manhole'].shape}")  # [N1, 1]
print(f"Cell predictions: {pred_dict['cell'].shape}")  # [N2, 1]
```

### 3. Visualize Graph Topology

Use the existing `build_static.py`:
```bash
python build_static.py
```

## 🎯 Competition Submission Format

The submission CSV must have columns:
```
row_id, model_id, event_id, node_type, node_id, water_level
```

Example:
```csv
row_id,model_id,event_id,node_type,node_id,water_level
0,2,3,1,50,233.3301
1,2,1,2,90,254.7810
2,2,4,1,100,210.9821
```

Where:
- `node_type=1`: Manhole (1D)
- `node_type=2`: Cell (2D)

## 🔬 Advanced Features

### 1. Multi-Model Ensemble

Train separate models for Model 1 and Model 2, then ensemble:
```python
# Train Model 1
train_config.model_id = 1
train(model_config, train_config)

# Train Model 2
train_config.model_id = 2
train(model_config, train_config)

# Average predictions at inference
```

### 2. Boundary Node Detection

The model implicitly learns outlet behavior through:
- **Topology**: Out-degree analysis from `edge_index`
- **Dynamics**: Temporal water level patterns (outlets drain faster)
- **Attention**: GATv2Conv learns to attend to boundary nodes

### 3. Feature Engineering

Consider adding:
- **Temporal embeddings**: Sinusoidal encoding of time
- **Spatial embeddings**: Node coordinates normalized
- **Derived features**: Flow velocity, volume changes

## 📝 Citation

If you use this pipeline, please cite:
```bibtex
@misc{urbanfloodbench2026,
  title={UrbanFloodBench GNN Pipeline},
  author={Your Name},
  year={2026},
  howpublished={Kaggle Competition}
}
```

## 🤝 Contributing

Improvements welcome! Key areas:
- [ ] Add attention visualization
- [ ] Implement mixed precision training (AMP)
- [ ] Add learning rate finder
- [ ] Implement uncertainty quantification

## 📄 License

See competition rules and data license.

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `hidden_dim` or `batch_size`
- Process events sequentially instead of batching
- Enable gradient checkpointing

### Poor Convergence
- Lower learning rate
- Increase `num_recurrent_steps`
- Add more GNN layers
- Tune teacher forcing schedule

### NaN Losses
- Check for division by zero in standardization
- Enable gradient clipping
- Verify input data has no NaNs
