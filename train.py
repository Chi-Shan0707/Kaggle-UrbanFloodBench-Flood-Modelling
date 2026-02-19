"""Training script for HeteroFloodGNN with autoregressive forecasting.

This script implements:
1. Teacher forcing with scheduled decay
2. Standardized RMSE loss
3. Autoregressive prediction during validation/test
4. Checkpointing and logging
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# torch_scatter 已用 PyTorch 原生 scatter_add 替代，无需额外安装
# 原: from torch_scatter import scatter
def scatter(src: torch.Tensor, index: torch.Tensor, dim: int,
            dim_size: int, reduce: str = 'sum') -> torch.Tensor:
    """torch_scatter.scatter 的纯 PyTorch 等价实现（仅支持 reduce='sum'）.

    等价关系:
        scatter(src, index, dim=0, dim_size=N, reduce='sum')
        → output[i] = Σ src[k]  for all k where index[k] == i

    这里使用 Tensor.scatter_add_ 实现，无需安装 torch_scatter 扩展。
    如需 reduce='mean'/'max' 等，可改用 torch.scatter_reduce（PyTorch 2.0+）。
    """
    if reduce != 'sum':
        raise NotImplementedError(f"Only reduce='sum' is supported, got '{reduce}'")
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, index, src)
    return out
from typing import Dict, Tuple
import logging
from tqdm import tqdm
from pathlib import Path

from dataset import UrbanFloodDataset
from model import HeteroFloodGNN
from config import ModelConfig, TrainingConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StandardizedRMSELoss(nn.Module):
    """Compute Standardized RMSE loss for predictions.
    
    RMSE is computed separately for each feature and node type, normalized by
    pre-computed standard deviations.
    """
    
    def __init__(self, std_manhole: torch.Tensor, std_cell: torch.Tensor):
        super().__init__()
        self.register_buffer('std_manhole', std_manhole)
        self.register_buffer('std_cell', std_cell)
    
    def forward(self, pred_dict: Dict[str, torch.Tensor], 
                target_dict: Dict[str, torch.Tensor],
                mask_dict: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_dict: {'manhole': [N1, D1], 'cell': [N2, D2]}
            target_dict: {'manhole': [N1, D1], 'cell': [N2, D2]}
            mask_dict: Optional masks for valid nodes
        
        Returns:
            Physics-aware RMSE: Ignores rainfall prediction for cells
        """
        losses = []
        
        # Manhole: Calculate Loss on ALL features (water_level, inlet_flow)
        pred_man = pred_dict['manhole']  # [N1, D1=2]
        target_man = target_dict['manhole']  # [N1, D1=2]
        if mask_dict and 'manhole' in mask_dict:
            mask = mask_dict['manhole']
            pred_man = pred_man[mask]
            target_man = target_man[mask]
        
        # Normalize by std and compute RMSE: [D1]
        diff_man = (pred_man - target_man) / (self.std_manhole + 1e-6)
        loss_man = torch.sqrt(torch.mean(diff_man ** 2, dim=0)).mean()
        losses.append(loss_man)
        
        # Cell: Calculate Loss ONLY on water_level(1) and water_volume(2)
        # IGNORE rainfall(0) - it's a known external forcing
        pred_cell = pred_dict['cell']  # [N2, D2=3]
        target_cell = target_dict['cell']  # [N2, D2=3]
        if mask_dict and 'cell' in mask_dict:
            mask = mask_dict['cell']
            pred_cell = pred_cell[mask]
            target_cell = target_cell[mask]
        
        # Slice [:, 1:] to skip rainfall (index 0)
        pred_cell_interest = pred_cell[:, 1:]  # [N2, 2] - water_level, water_volume
        target_cell_interest = target_cell[:, 1:]  # [N2, 2]
        std_cell_interest = self.std_cell[1:]  # [2] - std for level and volume only
        
        diff_cell = (pred_cell_interest - target_cell_interest) / (std_cell_interest + 1e-6)
        loss_cell = torch.sqrt(torch.mean(diff_cell ** 2, dim=0)).mean()
        losses.append(loss_cell)
        
        # Average across node types (equal weight)
        return torch.stack(losses).mean()

def compute_stats_from_events(dataset: UrbanFloodDataset, event_list: list) -> Dict[str, torch.Tensor]:
    """计算动态特征的均值和标准差."""
    logger.info("Computing mean/std from training data...")
    
    man_data_list = []
    cell_data_list = []
    
    for event_name in tqdm(event_list, desc="Loading stats"):
        event_data = dataset.load_event(event_name)
        # Flatten time and nodes dimensions: [T * N, D]
        # 直接使用，因为 event_data['manhole'] 已经是 Tensor 了
        man_data_list.append(event_data['manhole'].reshape(-1, event_data['manhole'].shape[-1]))
        cell_data_list.append(event_data['cell'].reshape(-1, event_data['cell'].shape[-1]))
    
    # Concat all data
    all_man = torch.cat(man_data_list, dim=0) # [Total_Samples, D1]
    all_cell = torch.cat(cell_data_list, dim=0) # [Total_Samples, D2]
    
    stats = {
        'man_mean': all_man.mean(dim=0),
        'man_std': all_man.std(dim=0),
        'cell_mean': all_cell.mean(dim=0),
        'cell_std': all_cell.std(dim=0)
    }
    
    logger.info(f"Stats computed. Manhole Water Std: {stats['man_std'][0]:.4f}")
    return stats

def compute_std_from_events(dataset: UrbanFloodDataset, event_list: list, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute standard deviations of ALL dynamic features across training events."""
    logger.info("Computing standard deviations from training data...")
    
    manhole_data = []
    cell_data = []
    
    for event_name in tqdm(event_list, desc="Loading events for std"):
        event_data = dataset.load_event(event_name)
        
        # manhole_dyn: [T, N1, D1=2] -> flatten to [T*N1, D1]
        manhole_data.append(event_data['manhole'].reshape(-1, 2).numpy())
        
        # cell_dyn: [T, N2, D2=3] -> flatten to [T*N2, D2]
        cell_data.append(event_data['cell'].reshape(-1, 3).numpy())
    
    # Concatenate all data and compute std per feature
    all_manhole = np.concatenate(manhole_data, axis=0)  # [Total, D1]
    all_cell = np.concatenate(cell_data, axis=0)  # [Total, D2]
    
    std_manhole = torch.from_numpy(all_manhole.std(axis=0).astype(np.float32))  # [D1]
    std_cell = torch.from_numpy(all_cell.std(axis=0).astype(np.float32))  # [D2]
    
    logger.info(f"Computed std_manhole={std_manhole.tolist()}, std_cell={std_cell.tolist()}")
    
    return std_manhole, std_cell

def get_teacher_forcing_ratio(epoch: int, config: TrainingConfig) -> float:
    """Compute teacher forcing ratio with linear decay."""
    if epoch >= config.teacher_forcing_decay_epochs:
        return config.teacher_forcing_ratio_end
    
    progress = epoch / config.teacher_forcing_decay_epochs
    ratio = config.teacher_forcing_ratio_start + progress * (
        config.teacher_forcing_ratio_end - config.teacher_forcing_ratio_start
    )
    return ratio


def physics_mass_conservation_loss(
        pred_cell_vol:  torch.Tensor,   # [N2]   模型预测的 t+1 时刻单元水量
        curr_cell_vol:  torch.Tensor,   # [N2]   当前 t 时刻单元水量（真实值）
        rainfall:       torch.Tensor,   # [N2]   t+1 时刻降雨带来的体积增量
        edge_index:     torch.Tensor,   # [2, E] cell→cell 有向边索引
        edge_flow:      torch.Tensor,   # [E, 1] 模型预测的边流量（有正负）
        dt:             float = 1.0,    # 时间步长（无量纲，默认 1）
) -> torch.Tensor:
    """基于浅水方程局部质量守恒的物理损失函数。

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【物理背景 - 浅水方程连续性方程（质量守恒）】
    对于任意一个离散控制体（2D 网格单元 i），其水量变化满足：

        ΔV_i = V_{i,t+1} - V_{i,t}
             = ( Σ_{j→i} Q_{ji} - Σ_{i→k} Q_{ik} ) × Δt + R_i

    其中：
      - V_i         = 单元 i 的水量 (m³)
      - Q_{ji}      = 从单元 j 流 **入** 单元 i 的流量 (m³/s)，正值
      - Q_{ik}      = 从单元 i 流 **出** 到单元 k 的流量 (m³/s)，正值
      - Δt          = 时间步长 (s)
      - R_i         = 降雨净补给量 (m³)，已经是体积增量

    化简为：
        ΔV_i = (流入_i - 流出_i) × Δt + R_i   ← 核心公式

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【为何对 edge_flow 使用 ReLU？】
    模型输出的 edge_flow 是**有符号**标量：
      - 正值 (+Q) → 水从 src → dst 方向流动（正方向）
      - 负值 (-Q) → 水从 dst → src 方向流动（反方向）

    我们希望"流入"和"流出"都是非负物理量，因此：
      - flow_pos = ReLU(Q)  ≥ 0  → 仅保留 src→dst 方向的正流量
      - 流入节点 j:  scatter(flow_pos, edge_index[1], reduce='sum') → 汇聚到目标节点
      - 流出节点 i:  scatter(flow_pos, edge_index[0], reduce='sum') → 从源节点扣除
    这样反向流（负值）不会被错误地计入正方向的流入/流出，
    避免流量方向混乱导致质量守恒公式两侧符号对消。
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Args:
        pred_cell_vol:  模型预测的 t+1 时刻水量，形状 [N2]
        curr_cell_vol:  当前 t 时刻水量（真实值），形状 [N2]
        rainfall:       t+1 时刻降雨体积增量，形状 [N2]
        edge_index:     cell→cell 有向边索引，形状 [2, E]
        edge_flow:      模型预测的有向边流量（带符号），形状 [E, 1]
        dt:             时间步长，默认 1.0

    Returns:
        physics_loss:   L1 距离（标量），衡量预测水量变化与物理守恒约束的偏差
    """
    N2 = pred_cell_vol.shape[0]         # 节点总数（2D cell 数量）
    src = edge_index[0]                  # [E] 源节点索引（流出方）
    dst = edge_index[1]                  # [E] 目标节点索引（流入方）

    # ── Step 1: 提取有方向的正流量 ──────────────────────────────────────────
    # edge_flow 形状 [E, 1]，先 squeeze 为 [E]
    # ReLU 仅保留正向流量（src→dst），负值（反向）置 0
    # 如果不用 ReLU，负流量会被误当作"流出减少"或"流入减少"，
    # 导致物理公式 ΔV = 流入 - 流出 + R 计算错误
    flow_pos = torch.relu(edge_flow.squeeze(-1))  # [E]，非负，单位与 edge_flow 相同

    # ── Step 2: 节点级流入量聚合 ─────────────────────────────────────────────
    # 对每条边的目标节点（dst）做 sum 聚合：汇总所有流入该节点的流量
    # scatter_add 原生 PyTorch 实现，等价于 torch_scatter.scatter(reduce='sum')
    # → 输出 [N2]，index i 处 = Σ flow_pos[e]  for all e where dst[e] == i
    inflow = scatter(flow_pos, dst, dim=0, dim_size=N2, reduce='sum')   # [N2]

    # ── Step 3: 节点级流出量聚合 ─────────────────────────────────────────────
    # 对每条边的源节点（src）做 sum 聚合：汇总所有从该节点流出的流量
    outflow = scatter(flow_pos, src, dim=0, dim_size=N2, reduce='sum')  # [N2]

    # ── Step 4: 物理约束项（连续性方程）──────────────────────────────────────
    # 核心公式：ΔV_物理 = (流入 - 流出) × Δt + 降雨补给
    # 单位校验：[m³/s] × [s] + [m³] = [m³]  ✓（假设 edge_flow 单位 m³/s）
    delta_v_physics = (inflow - outflow) * dt + rainfall   # [N2]

    # ── Step 5: 模型预测的水量变化 ───────────────────────────────────────────
    # 对应的模型预测水量变化：ΔV_pred = V_{t+1,pred} - V_{t,true}
    # 使用 V_{t,true} 作为基准，避免累积误差干扰物理约束的梯度方向
    delta_v_pred = pred_cell_vol - curr_cell_vol             # [N2]

    # ── Step 6: 物理损失（L1） ────────────────────────────────────────────────
    # 用 L1（绝对误差均值）而非 MSE：
    #   - 对离群流量值更鲁棒（洪水峰值时流量可能极大）
    #   - 梯度恒为 ±1（不受误差量级放大影响），训练更稳定
    physics_loss = torch.abs(delta_v_pred - delta_v_physics).mean()  # 标量

    return physics_loss


def train_one_event(model: HeteroFloodGNN, data, event_data: Dict,
                   criterion: nn.Module, optimizer: torch.optim.Optimizer,
                   device: str,
                   teacher_forcing_ratio: float = 1.0,
                   physics_loss_weight: float = 0.1,
                   chunk_size: int = 20,
                   gradient_clip: float = 1.0) -> float:
    """Train on one event using Truncated BPTT with chunked multi-step rollout loss.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【截断沿时间反向传播 (Truncated BPTT) — 设计动机】

    问题 1 — 传统逐步反传（原始实现）：
        原来每个时间步都调用 loss.backward() 并立即 detach h_dict。
        这等价于把序列模型退化成"只看 1 步"的前馈网络：
          梯度永远传不过当前时刻，模型无法学习跨步的时间依赖关系。

    问题 2 — 完整 BPTT（T 步全部展开）：
        把整个序列（T ≈ 400-600 步）的计算图保留在内存中再反传，
        对于 GNN 模型而言会导致显存 OOM（Out-Of-Memory），不可行。

    解决方案 — Truncated BPTT（截断 BPTT）：
        将长序列切分为若干长度为 chunk_size（默认 20）的子段。
        在每个子段内：
          1. 完整展开计算图，累积 chunk_size 步的损失。
          2. 对累积损失做一次 backward（梯度在 chunk 内完整回传）。
          3. 调用 optimizer.step() 更新权重。
          4. 调用 h_dict.detach() 切断子段间的计算图（防止 OOM）。
          5. 重置累积损失，进入下一个 chunk。
        这样梯度可以在 chunk 内部跨越多个时间步，
        同时每次只保留 chunk_size 步的计算图，显存可控。

    形象比喻：把一部 500 帧的电影分成每段 20 帧分析，每段学完就释放胶片，
    而不是把整部电影全部撑在内存里，也不是每帧都单独"看一眼就忘"。
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Args:
        model:                HeteroFloodGNN 模型
        data:                 HeteroData 静态图
        event_data:           事件数据字典，含 'manhole' [T,N1,D1]、'cell' [T,N2,D2]
        criterion:            主损失函数（StandardizedRMSELoss）
        optimizer:            优化器（外部传入，内部负责 zero_grad + step）
        device:               torch device
        teacher_forcing_ratio: 本 epoch 的 teacher forcing 概率
        physics_loss_weight:  物理守恒损失权重 λ
        chunk_size:           每个 Truncated BPTT 子段的时间步数（默认 20）
        gradient_clip:        梯度裁剪阈值（防梯度爆炸）

    Returns:
        avg_loss: 该 event 所有时间步的平均损失（float，用于日志）
    """
    model.train()

    manhole_seq = event_data['manhole'].to(device)  # [T, N1, D1]
    cell_seq    = event_data['cell'].to(device)      # [T, N2, D2]
    T = manhole_seq.shape[0]

    # 静态拓扑边索引（全序列不变）
    cell_edge_index = data['cell', 'to_cell', 'cell'].edge_index  # [2, E_2d]

    total_loss     = 0.0          # 用于日志记录（纯 Python float，无计算图）
    steps_counted  = 0            # 已统计的时间步数

    # ── 截断 BPTT 核心状态 ──────────────────────────────────────────────────
    # accumulated_loss: 当前 chunk 内累积的可微分损失张量
    # h_dict:           跨 chunk 传递的隐状态（在 chunk 边界被 detach）
    accumulated_loss = None       # 尚未开始累积时为 None
    h_dict = None

    # ── 每个 chunk 开始前清零梯度 ──────────────────────────────────────────
    # 注意：zero_grad 在每个 chunk 开始前（即每次 optimizer.step() 之后）调用，
    # 而不是在整个 event 开始前调用一次。这样可以保证每个 chunk 的梯度互相独立。
    optimizer.zero_grad()

    for t in range(T - 1):

        # ── 前向传播（单步）──────────────────────────────────────────────────
        manhole_dyn_t = manhole_seq[t]  # [N1, D1]
        cell_dyn_t    = cell_seq[t]     # [N2, D2]

        pred_dict, h_dict, edge_flow = model(data, manhole_dyn_t, cell_dyn_t, h_dict)

        # ── 计算本步损失 ──────────────────────────────────────────────────────
        target_dict = {
            'manhole': manhole_seq[t+1],
            'cell':    cell_seq[t+1],
        }

        rmse_loss = criterion(pred_dict, target_dict)

        pred_cell_vol = pred_dict['cell'][:, 2]
        curr_cell_vol = cell_dyn_t[:, 2].detach()
        rainfall_t1   = cell_seq[t+1, :, 0].detach()

        p_loss = physics_mass_conservation_loss(
            pred_cell_vol = pred_cell_vol,
            curr_cell_vol = curr_cell_vol,
            rainfall      = rainfall_t1,
            edge_index    = cell_edge_index,
            edge_flow     = edge_flow,
            dt            = 1.0,
        )

        step_loss = rmse_loss + physics_loss_weight * p_loss

        # ── 将本步损失累加到当前 chunk ───────────────────────────────────────
        # 使用加法（而非 .item()）保留计算图，使梯度能回传到模型参数
        if accumulated_loss is None:
            accumulated_loss = step_loss
        else:
            accumulated_loss = accumulated_loss + step_loss

        total_loss    += step_loss.item()
        steps_counted += 1

        # ── Teacher Forcing：决定下一步的输入来源 ───────────────────────────
        with torch.no_grad():
            if np.random.rand() >= teacher_forcing_ratio:
                # 使用模型预测值作为下一步输入（自回归），并保留真实降雨
                manhole_seq[t+1] = pred_dict['manhole'].detach()
                next_cell        = pred_dict['cell'].detach()
                next_cell[:, 0]  = cell_seq[t+1, :, 0].clone()
                cell_seq[t+1]    = next_cell

        # ── Truncated BPTT 边界判断 ──────────────────────────────────────────
        # 当本 chunk 恰好积累了 chunk_size 步，或者到达序列末尾时触发一次反传。
        #
        # 【为什么在这里 backward + detach，而不是每步都 backward？】
        # ① 在 chunk 内：计算图连通，梯度可以从第 chunk_size 步反向流回第 1 步，
        #    让模型学到 chunk_size 步的时间依赖。
        # ② chunk 边界 detach：切断子段间的计算图，下一个 chunk 的前向传播
        #    从一个"叶节点"隐状态出发，不会回溯更早的历史，避免显存爆炸。
        # ③ 每个 chunk 做一次 optimizer.step()：权重在每个 chunk 后更新，
        #    相当于在一个 event 内做多次小梯度更新，训练更细粒度、更稳定。
        is_chunk_end   = ((t + 1) % chunk_size == 0)
        is_seq_end     = (t == T - 2)

        if is_chunk_end or is_seq_end:
            # ① 归一化：把 chunk 内累积损失除以实际步数，确保不同长度 chunk 的梯度量级一致
            actual_chunk_steps = (steps_counted % chunk_size) or chunk_size
            chunk_loss_normed  = accumulated_loss / actual_chunk_steps

            # ② 反向传播：梯度在本 chunk 内完整展开
            chunk_loss_normed.backward()

            # ③ 梯度裁剪：防止长依赖导致梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            # ④ 更新参数
            optimizer.step()

            # ⑤ 清零梯度，为下一个 chunk 做准备
            optimizer.zero_grad()

            # ⑥ 复位累积损失（释放当前 chunk 的计算图，节省显存）
            accumulated_loss = None

            # ⑦ 截断隐状态的计算图（Truncation）
            # 这是 Truncated BPTT 的核心操作：
            # h_dict 仍然携带"数值"（序列状态信息保留），
            # 但梯度不再能穿越这个点向更早的时间步传播（计算图被切断）。
            # 若不 detach，PyTorch 会尝试跨 chunk 反传，
            # 最终导致 T 步全展开的超长计算图，触发 OOM。
            if not is_seq_end:
                h_dict = {k: v.detach() for k, v in h_dict.items()}

    return total_loss / max(steps_counted, 1)


@torch.no_grad()
def validate_one_event(model: HeteroFloodGNN, data, event_data: Dict,
                      criterion: nn.Module, device: str) -> float:
    """Validate on one event with full autoregressive prediction.
    
    No teacher forcing - always use model predictions.
    """
    model.eval()
    
    manhole_seq = event_data['manhole'].to(device)
    cell_seq = event_data['cell'].to(device)
    T = manhole_seq.shape[0]
    
    total_loss = 0.0
    h_dict = None
    
    # Use ground truth for the first step only
    manhole_dyn_t = manhole_seq[0].clone()
    cell_dyn_t = cell_seq[0].clone()
    
    for t in range(T - 1):
        # forward 返回 3-tuple: (pred_dict, h_dict, cell_to_cell_flow)；验证阶段暂不使用边流量
        pred_dict, h_dict, _ = model(data, manhole_dyn_t, cell_dyn_t, h_dict)
        
        target_dict = {
            'manhole': manhole_seq[t+1],  # [N1, D1] - all features
            'cell': cell_seq[t+1]  # [N2, D2] - all features
        }
        
        loss = criterion(pred_dict, target_dict)
        total_loss += loss.item()
        
        # Always use predictions for next step BUT inject true rainfall
        manhole_dyn_t = pred_dict['manhole']  # [N1, D1]
        
        # Inject true rainfall for physical consistency
        cell_dyn_t = pred_dict['cell'].clone()  # [N2, D2]
        true_rainfall = cell_seq[t+1, :, 0]  # [N2]
        cell_dyn_t[:, 0] = true_rainfall
    
    return total_loss / (T - 1)


def train(model_config: ModelConfig, train_config: TrainingConfig):
    """Main training loop."""
    
    # Setup
    device = torch.device(train_config.device)
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset for Model {train_config.model_id}...")
    dataset = UrbanFloodDataset(root="./", model_id=train_config.model_id)
    data = dataset.get(0)

    # ==================== 【新增代码 START】 ====================
    # 计算并注入统计量
    logger.info("Injecting normalization stats into model...")
    stats = compute_stats_from_events(dataset, train_config.train_events)
    
    # Initialize model
    model = HeteroFloodGNN(
        config=model_config,
        manhole_static_dim=4,  # depth, invert, surface, area
        cell_static_dim=6,  # area, rough, min_elev, elev, aspect, curv
        manhole_dynamic_dim=2,  # water_level, inlet_flow
        cell_dynamic_dim=3  # rainfall, water_level, water_volume
    ).to(device)
    
    # 关键步骤：把计算出的 mean/std 复制到模型的 buffer 里
    # 这样它们就会变成模型的一部分，随 model.pt 保存
    model.man_dyn_mean.copy_(stats['man_mean'].to(device))
    model.man_dyn_std.copy_(stats['man_std'].to(device))
    model.cell_dyn_mean.copy_(stats['cell_mean'].to(device))
    model.cell_dyn_std.copy_(stats['cell_std'].to(device))
    # ==================== 【新增代码 END】 ====================

    # Compute standard deviations if needed
    if train_config.use_standardized_rmse:
        std_manhole, std_cell = compute_std_from_events(
            dataset, train_config.train_events, device
        )
        train_config.std_manhole = std_manhole
        train_config.std_cell = std_cell
    
   
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, 
                     weight_decay=train_config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.num_epochs)
    
    # Loss function
    criterion = StandardizedRMSELoss(train_config.std_manhole, train_config.std_cell)
    criterion = criterion.to(device)  # Move criterion buffers to device
    
    # Move data to device
    data = data.to(device)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(train_config.num_epochs):
        # Get teacher forcing ratio for this epoch
        tf_ratio = get_teacher_forcing_ratio(epoch, train_config)
        
        # Train
        model.train()
        train_losses = []
        
        for event_name in tqdm(train_config.train_events, desc=f"Epoch {epoch+1}/{train_config.num_epochs}"):
            event_data = dataset.load_event(event_name)

            # optimizer.zero_grad / backward / clip / step 全部由 train_one_event 内部
            # 的 Truncated BPTT 逻辑在每个 chunk 边界负责，此处不再重复调用
            loss = train_one_event(
                model, data, event_data, criterion, optimizer, device,
                teacher_forcing_ratio  = tf_ratio,
                physics_loss_weight    = 0.1,
                chunk_size             = 20,
                gradient_clip          = train_config.gradient_clip,
            )
            train_losses.append(loss)
        
        avg_train_loss = np.mean(train_losses)
        
        # Validate
        val_losses = []
        for event_name in train_config.val_events:
            event_data = dataset.load_event(event_name)
            val_loss = validate_one_event(model, data, event_data, criterion, device)
            val_losses.append(val_loss)
        
        avg_val_loss = np.mean(val_losses)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                   f"Val Loss={avg_val_loss:.4f}, TF Ratio={tf_ratio:.2f}, "
                   f"LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(train_config.checkpoint_dir) / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'model_config': model_config,
                'train_config': train_config,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        if (epoch + 1) % train_config.save_every_n_epochs == 0:
            checkpoint_path = Path(train_config.checkpoint_dir) / f"model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    # Example configuration
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
        learning_rate=1e-4,
        num_epochs=8,
        teacher_forcing_ratio_start=1.0,
        teacher_forcing_ratio_end=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    train(model_config, train_config)
