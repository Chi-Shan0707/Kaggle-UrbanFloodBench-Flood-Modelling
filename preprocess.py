import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse

def process_split(root_dir, model_id, split):
    """处理单个 split (train 或 test) 下的所有 event"""
    base_dir = os.path.join(root_dir, f"Models/Model_{model_id}/{split}")
    
    if not os.path.exists(base_dir):
        print(f"⚠️ 目录不存在，跳过: {base_dir}")
        return

    print(f"🚀 正在处理 Model {model_id} - {split} ...")

    # 1. 加载静态映射表 (用于将 node_id 映射到 0..N-1 索引)
    # 逻辑必须与 dataset.py 完全一致
    man_static_fp = os.path.join(base_dir, '1d_nodes_static.csv')
    cell_static_fp = os.path.join(base_dir, '2d_nodes_static.csv')
    
    man_df = pd.read_csv(man_static_fp)
    cell_df = pd.read_csv(cell_static_fp)
    
    # 建立映射字典
    man_map = {orig: i for i, orig in enumerate(man_df['node_idx'].astype(int))}
    cell_map = {orig: i for i, orig in enumerate(cell_df['node_idx'].astype(int))}
    
    N1, N2 = len(man_map), len(cell_map)
    
    # 2. 遍历所有 Event
    event_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and 'event' in d]
    
    for event in tqdm(event_folders, desc=f"转换 CSV -> PT"):
        event_path = os.path.join(base_dir, event)
        output_pt_path = os.path.join(event_path, 'event_data.pt')
        
        # 如果已经存在且不想覆盖，可以取消注释下面两行
        # if os.path.exists(output_pt_path):
        #     continue

        # --- 读取动态数据 (逻辑复刻自 dataset.py) ---
        man_dyn_fp = os.path.join(event_path, '1d_nodes_dynamic_all.csv')
        cell_dyn_fp = os.path.join(event_path, '2d_nodes_dynamic_all.csv')
        timesteps_fp = os.path.join(event_path, 'timesteps.csv')
        
        # 快速检查文件
        if not (os.path.exists(man_dyn_fp) and os.path.exists(cell_dyn_fp)):
            print(f"❌ 缺少文件，跳过: {event}")
            continue

        man_dyn = pd.read_csv(man_dyn_fp)
        cell_dyn = pd.read_csv(cell_dyn_fp)
        ts = pd.read_csv(timesteps_fp)
        T = len(ts)

        # -------------------------------------------------
        # 核心转换逻辑 (从 Pandas GroupBy 转为 Numpy Array)
        # -------------------------------------------------
        
        # 1. Manhole: [water_level, inlet_flow]
        man_feat_cols = ['water_level', 'inlet_flow']
        D1 = len(man_feat_cols)
        man_tensor = np.zeros((T, N1, D1), dtype=np.float32)
        
        for t_idx, group in man_dyn.groupby('timestep'):
            # 映射 node_idx -> tensor index
            node_indices = group['node_idx'].map(man_map).values
            valid_mask = ~np.isnan(node_indices)
            valid_indices = node_indices[valid_mask].astype(int)
            
            # 填入数据
            values = group[man_feat_cols].values[valid_mask]
            man_tensor[int(t_idx), valid_indices, :] = values

        # 2. Cell: [rainfall, water_level, water_volume]
        cell_feat_cols = ['rainfall', 'water_level', 'water_volume']
        D2 = len(cell_feat_cols)
        cell_tensor = np.zeros((T, N2, D2), dtype=np.float32)

        for t_idx, group in cell_dyn.groupby('timestep'):
            node_indices = group['node_idx'].map(cell_map).values
            valid_mask = ~np.isnan(node_indices)
            valid_indices = node_indices[valid_mask].astype(int)
            
            values = group[cell_feat_cols].values[valid_mask]
            cell_tensor[int(t_idx), valid_indices, :] = values
            
        # -------------------------------------------------
        # 保存为 .pt 文件
        # -------------------------------------------------
        save_dict = {
            'manhole': torch.from_numpy(man_tensor), # 转为 Tensor
            'cell': torch.from_numpy(cell_tensor),   # 转为 Tensor
            'timesteps': T
        }
        
        torch.save(save_dict, output_pt_path)

if __name__ == "__main__":
    # 可以在这里指定要处理的模型 ID
    # 处理 Model 1 和 Model 2 的 train 和 test 数据
    for mid in [1, 2]:
        for mode in ['train', 'test']:
            process_split(root_dir="./", model_id=mid, split=mode)
            
    print("\n✅ 所有数据预处理完成！现在请修改 dataset.py 使用 .pt 文件。")