from typing import Optional, List, Dict
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, HeteroData

class UrbanFloodDataset(InMemoryDataset):
    """
    Spatio-Temporal Heterogeneous Graph Dataset for Urban Flood Forecasting.
    Updates: Added 'split' support (train/test) to fix path issues.
    """

    def __init__(self, root: str, model_id: int = 2, split: str = 'train', 
                 transform=None, pre_transform=None):
        self.root = root
        self.model_id = model_id
        self.split = split.lower() # 确保是 'train' 或 'test'
        
        # 必须在 super().__init__ 之前设置好 split，因为 raw_dir 和 processed_file_names 依赖它
        super().__init__(root, transform, pre_transform)
        
        # 加载处理好的数据，显式允许加载复杂对象
        # 注意：这里会自动根据 split 加载对应的 .pt 文件
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        # 【关键修改】动态指向 train 或 test 目录
        return os.path.join(self.root, f"Models/Model_{self.model_id}/{self.split}")

    @property
    def raw_file_names(self) -> List[str]:
        # 返回空列表以跳过 PyG 的自动下载/检查，我们手动管理 raw_dir
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        # 【关键修改】处理后的文件名包含 split，防止训练集覆盖测试集图结构
        # 例如: model_2_train_graph.pt 或 model_2_test_graph.pt
        return [f'model_{self.model_id}_{self.split}_graph.pt']

    def process(self):
        # 1. 检查原始文件路径 (此时 raw_dir 已经根据 split 变化了)
        base = self.raw_dir
        manhoe_static_fp = os.path.join(base, '1d_nodes_static.csv')
        cell_static_fp = os.path.join(base, '2d_nodes_static.csv')
        e1_fp = os.path.join(base, '1d_edge_index.csv')
        e2_fp = os.path.join(base, '2d_edge_index.csv')
        cpl_fp = os.path.join(base, '1d2d_connections.csv')

        print(f"Processing raw data from: {base}")
        
        # 简单检查文件是否存在
        if not os.path.exists(manhoe_static_fp):
            raise FileNotFoundError(f"Static files not found in {base}. Please check if the directory exists.")

        # 2. 读取 CSV
        man_df = pd.read_csv(manhoe_static_fp)
        cell_df = pd.read_csv(cell_static_fp)

        # 3. 建立 ID 映射 (Node Index -> 0..N-1)
        man_idx = man_df['node_idx'].astype(int).values
        cell_idx = cell_df['node_idx'].astype(int).values
        man_map = {orig: i for i, orig in enumerate(man_idx)}
        cell_map = {orig: i for i, orig in enumerate(cell_idx)}

        # 4. 提取静态特征
        man_static_cols = ['depth', 'invert_elevation', 'surface_elevation', 'base_area']
        man_static = man_df[man_static_cols].fillna(0).astype(np.float32).values

        cell_static_cols = ['area', 'roughness', 'min_elevation', 'elevation', 'aspect', 'curvature']
        cell_static = cell_df[cell_static_cols].fillna(0).astype(np.float32).values

        # 5. 构建 HeteroData 对象
        data = HeteroData()
        data['manhole'].x_static = torch.from_numpy(man_static)
        data['cell'].x_static = torch.from_numpy(cell_static)

        # 6. 构建边索引 (Edge Index)
        e1 = pd.read_csv(e1_fp)
        e2 = pd.read_csv(e2_fp)

        def build_edge_index(df, src_map, dst_map):
            src = df['from_node'].astype(int).map(src_map).values
            dst = df['to_node'].astype(int).map(dst_map).values
            # 过滤掉无法映射的边
            valid = (~np.isnan(src)) & (~np.isnan(dst))
            edge_index = np.vstack([src[valid], dst[valid]]).astype(np.int64)
            return torch.from_numpy(edge_index)

        data['manhole', 'to_manhole', 'manhole'].edge_index = build_edge_index(e1, man_map, man_map)
        data['cell', 'to_cell', 'cell'].edge_index = build_edge_index(e2, cell_map, cell_map)

        # 7. 构建 1D-2D 耦合边
        cpl = pd.read_csv(cpl_fp)
        src_1d = cpl['node_1d'].astype(int).map(man_map).values
        dst_2d = cpl['node_2d'].astype(int).map(cell_map).values
        
        valid_cpl = (~np.isnan(src_1d)) & (~np.isnan(dst_2d))
        src_1d = src_1d[valid_cpl]
        dst_2d = dst_2d[valid_cpl]

        # 双向连接
        data['manhole', 'to_cell', 'cell'].edge_index = torch.from_numpy(np.vstack([src_1d, dst_2d]).astype(np.int64))
        data['cell', 'to_manhole', 'manhole'].edge_index = torch.from_numpy(np.vstack([dst_2d, src_1d]).astype(np.int64))

        # 8. 保存原始 ID 以备后用
        data['manhole'].orig_idx = torch.from_numpy(man_idx.astype(np.int64))
        data['cell'].orig_idx = torch.from_numpy(cell_idx.astype(np.int64))

        # 9. 内存化与保存
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.data, self.slices = self.collate([data])
        
        # 创建 processed 目录并保存文件
        print(f"Saving processed graph to {self.processed_paths[0]}")
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def load_event(self, event_folder: str) -> Dict[str, torch.Tensor]:
        """Load dynamic sequences for a single event folder."""
        # 这里的 raw_dir 会根据 self.split 自动变为 .../train 或 .../test
        base = self.raw_dir
        
        # 重新加载映射表
        manhoe_static_fp = os.path.join(base, '1d_nodes_static.csv')
        cell_static_fp = os.path.join(base, '2d_nodes_static.csv')
        
        man_df = pd.read_csv(manhoe_static_fp)
        cell_df = pd.read_csv(cell_static_fp)
        
        man_map = {orig: i for i, orig in enumerate(man_df['node_idx'].astype(int))}
        cell_map = {orig: i for i, orig in enumerate(cell_df['node_idx'].astype(int))}
        
        ev_base = os.path.join(base, event_folder)
        man_dyn_fp = os.path.join(ev_base, '1d_nodes_dynamic_all.csv')
        cell_dyn_fp = os.path.join(ev_base, '2d_nodes_dynamic_all.csv')
        timesteps_fp = os.path.join(ev_base, 'timesteps.csv')

        if not os.path.exists(man_dyn_fp):
            raise FileNotFoundError(f"Dynamic file missing: {man_dyn_fp}")

        man_dyn = pd.read_csv(man_dyn_fp)
        cell_dyn = pd.read_csv(cell_dyn_fp)
        ts = pd.read_csv(timesteps_fp)
        T = len(ts)

        # Manhole dynamics
        man_feat_cols = ['water_level', 'inlet_flow']
        N1 = len(man_map)
        D1 = len(man_feat_cols)
        man_tensor = np.zeros((T, N1, D1), dtype=np.float32)
        
        for t_idx, group in man_dyn.groupby('timestep'):
            node_indices = group['node_idx'].map(man_map).values
            valid_mask = ~np.isnan(node_indices)
            valid_indices = node_indices[valid_mask].astype(int)
            values = group[man_feat_cols].values[valid_mask]
            man_tensor[int(t_idx), valid_indices, :] = values

        # Cell dynamics
        cell_feat_cols = ['rainfall', 'water_level', 'water_volume']
        N2 = len(cell_map)
        D2 = len(cell_feat_cols)
        cell_tensor = np.zeros((T, N2, D2), dtype=np.float32)

        for t_idx, group in cell_dyn.groupby('timestep'):
            node_indices = group['node_idx'].map(cell_map).values
            valid_mask = ~np.isnan(node_indices)
            valid_indices = node_indices[valid_mask].astype(int)
            values = group[cell_feat_cols].values[valid_mask]
            cell_tensor[int(t_idx), valid_indices, :] = values

        return {
            'manhole': torch.from_numpy(man_tensor),
            'cell': torch.from_numpy(cell_tensor),
            'timesteps': T,
            'tstamp_df': ts,
        }