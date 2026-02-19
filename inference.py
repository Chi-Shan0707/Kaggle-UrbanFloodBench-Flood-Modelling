"""Inference script for generating Kaggle submission file.

推理逻辑说明
===========
每个 test event 的时间轴结构:

  [t=0 ... t=context_len-1]  完整数据（真实 water_level/inlet_flow 已知）
  [t=context_len ... t=T-1]  缺失数据（water_level/water_volume 为 NaN，需预测）

推理分两阶段:
  Phase 1 – Context Warmup（t = 0 .. context_len-2）
    - 以真实数据作为输入，逐步更新 GRU 隐状态
    - 不收集预测结果（这些时间步是已知的，不需要提交）

  Phase 2 – Autoregressive Prediction（t = context_len-1 .. T-2）
    - 从最后一个已知时间步 (context_len-1) 出发
    - 每步将自身预测值滚动馈回作为下一步输入
    - 始终将真实 rainfall 注入 cell 特征的第 0 维（降雨是外部强迫，永远可知）
    - 收集每步的 water_level 预测值（manhole[:, 0], cell[:, 1]）

与训练的一致性
==============
✓ 残差预测（delta）: model.forward() 内部完成，推理侧无需额外处理
✓ 归一化 / 反归一化: model.forward() 内部通过 register_buffer 完成
✓ 边数据: 模型接受的是静态 edge_index（在 data 中），动态边流量是输出而非输入
✓ torch_scatter: 已替换为 PyTorch 原生 scatter_add_，推理阶段不涉及

输出格式
========
每行: row_id, model_id, event_id, node_type, node_id, timestep_idx, water_level
- timestep_idx: 该预测对应的时间步索引（context_len .. T-1），便于对齐与调试
- 按 event、timestep、node 升序写入，确保 make_submission.py 的顺序对齐正确
"""

import os
import gc
import logging
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from dataset import UrbanFloodDataset
from model import HeteroFloodGNN
from config import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _detect_context_len(manhole_seq: torch.Tensor) -> int:
    """检测每个 event 的 context 长度（非 NaN 的连续时间步数量）。

    Args:
        manhole_seq: [T, N1, D1] — manhole 动态特征序列，NaN 表示缺失目标时间步

    Returns:
        context_len: 从序列开头起连续非 NaN 的时间步数，至少为 1
    """
    # [T]: 每个时间步只要有任意 NaN 即视为缺失
    is_nan = manhole_seq[:, :, 0].isnan().any(dim=1)   # [T] bool
    nan_positions = is_nan.nonzero(as_tuple=False)
    if len(nan_positions) == 0:
        return manhole_seq.shape[0]   # 全部都是已知数据（训练集情况）
    return int(nan_positions[0].item())


# ─────────────────────────────────────────────────────────────────────────────
# 主推理函数
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_stream(checkpoint_path: str, model_id: int, output_csv: str, device):
    """逐 event 流式推理并写入 CSV（内存安全版本）。

    只输出「缺失时间步」（context_len .. T-1）的预测，不输出 warmup 期间的预测。
    """

    # ── 1. 加载模型 ──────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    model_config: ModelConfig = checkpoint['model_config']

    model = HeteroFloodGNN(
        config=model_config,
        manhole_static_dim=4,
        cell_static_dim=6,
        manhole_dynamic_dim=2,
        cell_dynamic_dim=3,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Model {model_id} loaded (epoch {checkpoint.get('epoch', 'N/A')})")

    # ── 2. 初始化数据集 ───────────────────────────────────────────────────────
    test_dir = Path(f"./Models/Model_{model_id}/test")
    split = 'test' if test_dir.exists() else 'train'
    if split == 'train':
        logger.warning("Test directory not found — falling back to TRAIN split!")

    dataset = UrbanFloodDataset(root="./", model_id=model_id, split=split)

    # 将静态图加载到推理设备
    static_data = dataset.get(0).to(device)

    # orig_idx 保存节点在原始 CSV 中的全局索引（即 submission 中的 node_id）
    orig_idx_1d = static_data['manhole'].orig_idx.cpu().numpy()  # [N1]
    orig_idx_2d = static_data['cell'].orig_idx.cpu().numpy()     # [N2]

    # ── 3. 收集 event 列表 ────────────────────────────────────────────────────
    raw_dir = Path(dataset.raw_dir)
    event_folders = sorted(
        [d.name for d in raw_dir.iterdir() if d.is_dir() and 'event' in d.name]
    )
    logger.info(f"Found {len(event_folders)} events under {raw_dir} ({split})")

    # ── 4. 跳过已处理的 events（断点续写）───────────────────────────────────
    processed_events: set = set()
    file_exists = os.path.exists(output_csv)
    if file_exists:
        try:
            df_meta = pd.read_csv(output_csv, usecols=['model_id', 'event_id'])
            existing = df_meta[df_meta['model_id'] == model_id]
            processed_events = set(existing['event_id'].unique())
            logger.info(f"Resuming: skipping {len(processed_events)} already-done events")
        except Exception:
            logger.warning("Could not read existing CSV; starting fresh.")

    # ── 5. 推理主循环 ─────────────────────────────────────────────────────────
    columns = ['row_id', 'model_id', 'event_id', 'node_type', 'node_id',
               'timestep_idx', 'water_level']
    total_rows_written = 0

    with open(output_csv, 'a') as f:
        if not file_exists:
            f.write(','.join(columns) + '\n')

        with torch.no_grad():
            for event_name in tqdm(event_folders, desc=f"Model {model_id}"):
                # 解析 event_id
                try:
                    event_id = int(event_name.split('_')[-1])
                except ValueError:
                    continue

                if event_id in processed_events:
                    continue

                # ── 加载 event 数据 ─────────────────────────────────────────
                try:
                    event_data = dataset.load_event(event_name)
                except Exception as e:
                    logger.error(f"Error loading {event_name}: {e}")
                    continue

                # 序列张量: [T, N, D]
                manhole_seq: torch.Tensor = event_data['manhole'].to(device)  # [T, N1, 2]
                cell_seq:    torch.Tensor = event_data['cell'].to(device)      # [T, N2, 3]
                T: int = manhole_seq.shape[0]

                # ── 检测 context 长度（动态，无需硬编码）────────────────────
                # 测试集所有 event 均为 10，但保持动态以应对边缘情况
                context_len = _detect_context_len(manhole_seq)
                pred_steps  = T - context_len   # 需要预测的时间步数

                if pred_steps <= 0:
                    logger.debug(f"{event_name}: all {T} steps are context, skip.")
                    continue

                logger.debug(
                    f"{event_name}: T={T}, context={context_len}, predict={pred_steps}"
                )

                # ── Phase 1: Context Warmup ────────────────────────────────
                # 用真实数据逐步建立 GRU 隐状态（t = 0 .. context_len-2）
                # 共运行 context_len-1 次前向传播；不收集预测输出
                h_dict = None
                for t in range(context_len - 1):
                    _, h_dict, _ = model(
                        static_data,
                        manhole_seq[t],   # [N1, 2] 真实值
                        cell_seq[t],      # [N2, 3] 真实值（含真实 rainfall）
                        h_dict,
                    )
                    # 截断隐状态计算图（推理无需梯度）
                    h_dict = {k: v.detach() for k, v in h_dict.items()}

                # ── Phase 2: Autoregressive Prediction ────────────────────
                # 从最后一个已知时间步 context_len-1 的真实值出发
                manhole_dyn_t = manhole_seq[context_len - 1].clone()  # [N1, 2]
                cell_dyn_t    = cell_seq[context_len - 1].clone()      # [N2, 3]

                event_rows = []

                for t in range(context_len - 1, T - 1):
                    # model.forward() 内部:
                    #   ① 归一化输入 (register_buffer man_dyn_mean/std, cell_dyn_mean/std)
                    #   ② Encoder → GRU-GNN (reset/update/candidate gates) → Decoder
                    #   ③ 残差预测: pred_norm = x_norm + delta_norm
                    #   ④ 反归一化输出到真实物理单位
                    # 返回 3-tuple; cell_to_cell_flow（边流量）推理时丢弃
                    pred_dict, h_dict, _ = model(
                        static_data,
                        manhole_dyn_t,
                        cell_dyn_t,
                        h_dict,
                    )
                    h_dict = {k: v.detach() for k, v in h_dict.items()}

                    # 该步预测对应的时间步索引（context_len .. T-1）
                    predicted_ts = t + 1

                    # 提取 water_level（提交所需特征）
                    # Manhole D1=[water_level, inlet_flow]      → index 0
                    # Cell    D2=[rainfall, water_level, water_volume] → index 1
                    man_wl  = pred_dict['manhole'][:, 0].cpu().numpy()   # [N1]
                    cell_wl = pred_dict['cell'][:, 1].cpu().numpy()      # [N2]

                    # 构建 CSV 行（1D 节点）
                    for idx, val in zip(orig_idx_1d, man_wl):
                        event_rows.append(
                            f"-1,{model_id},{event_id},1,{idx},{predicted_ts},{val:.6f}"
                        )
                    # 构建 CSV 行（2D 节点）
                    for idx, val in zip(orig_idx_2d, cell_wl):
                        event_rows.append(
                            f"-1,{model_id},{event_id},2,{idx},{predicted_ts},{val:.6f}"
                        )

                    # ── 准备下一步自回归输入 ────────────────────────────────
                    # Manhole: 直接使用模型预测（无可注入的已知特征）
                    manhole_dyn_t = pred_dict['manhole'].clone()     # [N1, 2]

                    # Cell: 使用模型预测，但注入已知的真实 rainfall
                    # 测试集中 rainfall（cell[:, 0]）在所有时间步均无 NaN
                    next_cell = pred_dict['cell'].clone()             # [N2, 3]
                    if predicted_ts < T:
                        # 用真实降雨替换模型预测的降雨（外部大气强迫，已知）
                        next_cell[:, 0] = cell_seq[predicted_ts][:, 0]
                    cell_dyn_t = next_cell

                # ── 写入磁盘 ────────────────────────────────────────────────
                if event_rows:
                    f.write('\n'.join(event_rows) + '\n')
                    total_rows_written += len(event_rows)

                # ── 清理显存 ────────────────────────────────────────────────
                del event_rows, manhole_seq, cell_seq, h_dict, pred_dict
                torch.cuda.empty_cache()
                gc.collect()

    logger.info(f"Inference complete. Total rows written: {total_rows_written:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 后处理：重置 row_id（可选）
# ─────────────────────────────────────────────────────────────────────────────

def post_process_submission(output_csv: str):
    """对推理输出文件重置 row_id（从 0 开始连续编号）。

    注意：此步骤会将整个文件读入内存。
    如果文件超过可用内存，请改用 make_submission.py 进行最终对齐。
    """
    logger.info("Post-processing: resetting row_id ...")
    try:
        df = pd.read_csv(output_csv)
        df['row_id'] = range(len(df))
        df.to_csv(output_csv, index=False)
        logger.info(f"Done. {len(df):,} rows saved to {output_csv}")
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        logger.warning("Submission exists but row_ids are -1. Use make_submission.py to align.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UrbanFloodBench inference — only predicts missing timesteps"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--model_id", type=int, required=True,
                        help="Target Model ID (1 or 2)")
    parser.add_argument("--output", default="submission.csv",
                        help="Output CSV path (appended if exists)")
    parser.add_argument("--postprocess", action="store_true",
                        help="Reset row_id to sequential integers after inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    run_inference_stream(args.checkpoint, args.model_id, args.output, device)

    if args.postprocess:
        post_process_submission(args.output)


if __name__ == "__main__":
    main()
