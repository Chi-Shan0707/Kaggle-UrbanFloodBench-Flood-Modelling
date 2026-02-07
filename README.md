
# UrbanFloodBench GNN 训练流水线

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

## 📁 代码结构

```
.
├── dataset.py          # 自定义 UrbanFloodDataset (HeteroData 构建, 修复了 pickle 问题)
├── model.py            # HeteroFloodGNN 模型定义 (修复了 GATv2Conv 异构边自环问题)
├── train.py            # 训练循环 (包含 Teacher Forcing 和 detach 梯度截断)
├── test_pipeline.py    # 单元测试脚本 (用于验证数据加载和模型前向传播)
├── config.py           # 超参数配置
└── README.md           # 本文件

```

## 🚀 快速开始

### 1. 环境准备

基于 PyTorch 2.9.1+cu130 (适配 RTX 5060/Blackwell):

```bash
# 核心环境
conda create -n floodenv python=3.10 -y
conda activate floodenv

# 安装依赖 (无需 GeoPandas)
pip install torch==2.9.1+cu130 --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
pip install torch_geometric
pip install pandas numpy tqdm

```

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

## 🔧 常见故障排除

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
