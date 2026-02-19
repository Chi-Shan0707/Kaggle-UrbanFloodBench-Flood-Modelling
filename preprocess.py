"""preprocess.py — 将所有 CSV 文件批量转换为 .pt 格式。

输出物：
  1. Models/Model_{id}/{split}/static_graph.pt
       包含图的完整静态结构（节点特征、边索引、边静态特征、orig_idx 映射）

  2. Models/Model_{id}/{split}/event_{N}/event_data.pt
       包含该事件的动态时序张量（节点 + 边），以及 timesteps 等元数据

使用方法：
  python preprocess.py                           # 处理 Model 1/2 的 train/test
  python preprocess.py --model_id 2 --split train   # 仅处理指定模型/分片
  python preprocess.py --overwrite               # 强制覆盖已存在的 .pt 文件

说明：
  - 不依赖 torch_geometric，纯 pandas + numpy + torch。
  - 与 dataset.py 的索引映射逻辑完全一致（以 node_idx 列为准）。
  - 已存在的 .pt 文件默认跳过；添加 --overwrite 强制覆盖。
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ─────────────────────────── 通用工具 ────────────────────────────────────────

def _build_map(df: pd.DataFrame, col: str = 'node_idx') -> dict:
    """将 DataFrame col 列的原始 ID 映射到 0-based 连续整数索引。"""
    return {int(v): i for i, v in enumerate(df[col].values)}


def _build_edge_index(df: pd.DataFrame, src_map: dict, dst_map: dict) -> torch.Tensor:
    """从 edge DataFrame 构建 [2, E] 整数 edge_index，自动过滤无法映射的行。"""
    srcs = df['from_node'].astype(int).map(src_map).values
    dsts = df['to_node'].astype(int).map(dst_map).values
    valid = (~np.isnan(srcs.astype(float))) & (~np.isnan(dsts.astype(float)))
    return torch.from_numpy(
        np.vstack([srcs[valid], dsts[valid]]).astype(np.int64)
    )  # [2, E]


def _safe_float_tensor(df: pd.DataFrame, cols: list) -> torch.Tensor:
    """提取指定列，NaN→0，返回 float32 Tensor。"""
    return torch.from_numpy(df[cols].fillna(0).astype(np.float32).values)


def _load_edge_dyn(csv_path: str, T: int, E: int) -> torch.Tensor:
    """加载边动态 CSV → Tensor [T, E, 2]（flow, velocity）。
    文件不存在时返回全零张量。
    """
    arr = np.zeros((T, E, 2), dtype=np.float32)
    if not os.path.exists(csv_path):
        return torch.from_numpy(arr)
    df = pd.read_csv(csv_path)
    for t_idx, grp in df.groupby('timestep'):
        idxs = grp['edge_idx'].values.astype(int)
        valid = (idxs >= 0) & (idxs < E)
        arr[int(t_idx), idxs[valid], :] = \
            grp[['flow', 'velocity']].values[valid].astype(np.float32)
    return torch.from_numpy(arr)


# ─────────────────────── 静态图处理 ─────────────────────────────────────────

def process_static_graph(base_dir: str, overwrite: bool = False) -> dict:
    """从 base_dir 下的 CSV 构建静态图并保存为 static_graph.pt。

    保存字典 key：
      man_static     [N1, 4]   manhole 节点静态特征
      man_orig_idx   [N1]      manhole 原始 node_idx（推理结果对齐用）
      cell_static    [N2, 6]   cell 节点静态特征
      cell_orig_idx  [N2]      cell 原始 node_idx
      man2man_ei     [2, E1]   1D→1D edge_index
      man2man_attr   [E1, 7]   1D edge 静态特征
      cell2cell_ei   [2, E2]   2D→2D edge_index
      cell2cell_attr [E2, 5]   2D edge 静态特征
      man2cell_ei    [2, C]    manhole→cell 耦合 edge_index
      cell2man_ei    [2, C]    cell→manhole 耦合 edge_index
      N1, N2, E1, E2           节点/边数量（int）
    """
    output_path = os.path.join(base_dir, 'static_graph.pt')
    if os.path.exists(output_path) and not overwrite:
        print(f"  ✅ static_graph.pt 已存在，跳过构建")
        return torch.load(output_path, weights_only=False)

    # ── 节点静态特征 ──────────────────────────────────────────────────────────
    man_df  = pd.read_csv(os.path.join(base_dir, '1d_nodes_static.csv'))
    cell_df = pd.read_csv(os.path.join(base_dir, '2d_nodes_static.csv'))

    man_map  = _build_map(man_df)
    cell_map = _build_map(cell_df)
    N1, N2   = len(man_map), len(cell_map)

    man_static  = _safe_float_tensor(
        man_df, ['depth', 'invert_elevation', 'surface_elevation', 'base_area'])
    cell_static = _safe_float_tensor(
        cell_df, ['area', 'roughness', 'min_elevation', 'elevation', 'aspect', 'curvature'])

    # ── 边索引 ────────────────────────────────────────────────────────────────
    e1_df  = pd.read_csv(os.path.join(base_dir, '1d_edge_index.csv'))
    e2_df  = pd.read_csv(os.path.join(base_dir, '2d_edge_index.csv'))
    cpl_df = pd.read_csv(os.path.join(base_dir, '1d2d_connections.csv'))

    man2man_ei   = _build_edge_index(e1_df, man_map, man_map)
    cell2cell_ei = _build_edge_index(e2_df, cell_map, cell_map)

    src_1d   = cpl_df['node_1d'].astype(int).map(man_map).values
    dst_2d   = cpl_df['node_2d'].astype(int).map(cell_map).values
    valid_c  = (~np.isnan(src_1d.astype(float))) & (~np.isnan(dst_2d.astype(float)))
    src_1d   = src_1d[valid_c].astype(np.int64)
    dst_2d   = dst_2d[valid_c].astype(np.int64)

    man2cell_ei = torch.from_numpy(np.vstack([src_1d, dst_2d]))
    cell2man_ei = torch.from_numpy(np.vstack([dst_2d, src_1d]))

    E1 = man2man_ei.shape[1]
    E2 = cell2cell_ei.shape[1]

    # ── 边静态特征（可选，文件不存在时全零）──────────────────────────────────
    def _edge_static(fname, cols):
        p = os.path.join(base_dir, fname)
        if not os.path.exists(p):
            return torch.zeros(0, len(cols), dtype=torch.float32)
        df = pd.read_csv(p).sort_values('edge_idx').reset_index(drop=True)
        return _safe_float_tensor(df, cols)

    man2man_attr   = _edge_static('1d_edges_static.csv',
        ['relative_position_x', 'relative_position_y',
         'length', 'diameter', 'shape', 'roughness', 'slope'])
    cell2cell_attr = _edge_static('2d_edges_static.csv',
        ['relative_position_x', 'relative_position_y',
         'face_length', 'length', 'slope'])

    # ── 保存 ──────────────────────────────────────────────────────────────────
    graph = {
        'man_static'    : man_static,
        'man_orig_idx'  : torch.from_numpy(man_df['node_idx'].astype(np.int64).values),
        'cell_static'   : cell_static,
        'cell_orig_idx' : torch.from_numpy(cell_df['node_idx'].astype(np.int64).values),
        'man2man_ei'    : man2man_ei,
        'man2man_attr'  : man2man_attr,
        'cell2cell_ei'  : cell2cell_ei,
        'cell2cell_attr': cell2cell_attr,
        'man2cell_ei'   : man2cell_ei,
        'cell2man_ei'   : cell2man_ei,
        'N1': N1, 'N2': N2, 'E1': E1, 'E2': E2,
    }
    torch.save(graph, output_path)
    print(f"  💾 static_graph.pt 已保存: N1={N1}, N2={N2}, E1={E1}, E2={E2}")
    return graph


# ─────────────────────── 动态事件处理 ────────────────────────────────────────

def process_event(event_path: str, man_map: dict, cell_map: dict,
                  E1: int, E2: int, overwrite: bool = False) -> None:
    """将单个 event 目录下的 CSV 转换为 event_data.pt。

    保存字典 key：
      manhole    [T, N1, 2]   water_level, inlet_flow
      cell       [T, N2, 3]   rainfall, water_level, water_volume
      1d_edges   [T, E1, 2]   flow, velocity
      2d_edges   [T, E2, 2]   flow, velocity
      timesteps  int           时间步数 T
      tstamp_df  DataFrame     原始 timesteps.csv（含 timestamp 列）
    """
    output_path = os.path.join(event_path, 'event_data.pt')
    if os.path.exists(output_path) and not overwrite:
        return

    man_dyn_fp  = os.path.join(event_path, '1d_nodes_dynamic_all.csv')
    cell_dyn_fp = os.path.join(event_path, '2d_nodes_dynamic_all.csv')
    ts_fp       = os.path.join(event_path, 'timesteps.csv')

    if not (os.path.exists(man_dyn_fp) and os.path.exists(cell_dyn_fp)):
        print(f"  ❌ 缺少动态文件，跳过: {os.path.basename(event_path)}")
        return

    ts_df = pd.read_csv(ts_fp)
    T     = len(ts_df)
    N1    = len(man_map)
    N2    = len(cell_map)

    # Manhole [T, N1, 2]
    man_arr = np.zeros((T, N1, 2), dtype=np.float32)
    for t_idx, grp in pd.read_csv(man_dyn_fp).groupby('timestep'):
        idxs  = grp['node_idx'].map(man_map).values
        valid = ~np.isnan(idxs.astype(float))
        man_arr[int(t_idx), idxs[valid].astype(int), :] = \
            grp[['water_level', 'inlet_flow']].values[valid].astype(np.float32)

    # Cell [T, N2, 3]
    cell_arr = np.zeros((T, N2, 3), dtype=np.float32)
    for t_idx, grp in pd.read_csv(cell_dyn_fp).groupby('timestep'):
        idxs  = grp['node_idx'].map(cell_map).values
        valid = ~np.isnan(idxs.astype(float))
        cell_arr[int(t_idx), idxs[valid].astype(int), :] = \
            grp[['rainfall', 'water_level', 'water_volume']].values[valid].astype(np.float32)

    torch.save({
        'manhole'   : torch.from_numpy(man_arr),
        'cell'      : torch.from_numpy(cell_arr),
        '1d_edges'  : _load_edge_dyn(
            os.path.join(event_path, '1d_edges_dynamic_all.csv'), T, E1),
        '2d_edges'  : _load_edge_dyn(
            os.path.join(event_path, '2d_edges_dynamic_all.csv'), T, E2),
        'timesteps' : T,
        'tstamp_df' : ts_df,
    }, output_path)


# ─────────────────────── 主处理函数 ─────────────────────────────────────────

def process_split(root_dir: str, model_id: int, split: str,
                  overwrite: bool = False) -> None:
    """处理单个 (model_id, split) 组合的所有数据。"""
    base_dir = os.path.join(root_dir, f'Models/Model_{model_id}/{split}')

    if not os.path.exists(base_dir):
        print(f"⚠️  目录不存在，跳过: {base_dir}")
        return

    print(f"\n🚀 Model {model_id} - {split}  ({base_dir})")

    # 1. 静态图（含边索引 + 边/节点静态特征）
    graph = process_static_graph(base_dir, overwrite=overwrite)
    N1, N2, E1, E2 = graph['N1'], graph['N2'], graph['E1'], graph['E2']
    man_map  = {int(v): i for i, v in enumerate(graph['man_orig_idx'].tolist())}
    cell_map = {int(v): i for i, v in enumerate(graph['cell_orig_idx'].tolist())}

    # 2. 所有 event 动态数据
    event_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('event_')
    ])

    for ev in tqdm(event_dirs, desc=f'  M{model_id}-{split} CSV→PT'):
        process_event(os.path.join(base_dir, ev),
                      man_map, cell_map, E1, E2, overwrite=overwrite)

    print(f"  ✅ {len(event_dirs)} 个 event 处理完毕。")


# ─────────────────────── CLI 入口 ────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预处理 CSV → PT 文件')
    parser.add_argument('--model_id', type=int, default=None,
                        help='指定模型 ID（不指定则处理 1 和 2）')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'test'],
                        help='指定 split（不指定则处理 train 和 test）')
    parser.add_argument('--root', type=str, default='./',
                        help='项目根目录（含 Models/ 子目录）')
    parser.add_argument('--overwrite', action='store_true',
                        help='强制覆盖已存在的 .pt 文件')
    args = parser.parse_args()

    model_ids = [args.model_id] if args.model_id else [1, 2]
    splits    = [args.split]    if args.split    else ['train', 'test']

    for mid in model_ids:
        for sp in splits:
            process_split(root_dir=args.root, model_id=mid,
                          split=sp, overwrite=args.overwrite)

    print('\n✅ 全部预处理完成！')
