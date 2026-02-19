"""dataset.py — UrbanFloodDataset

静态图来源优先级：
  1. {split}/static_graph.pt  ← preprocess.py 生成，直接读取，速度最快
  2. 若不存在，自动调用 process() 从 CSV 重建（首次运行时）

动态事件来源优先级（load_event）：
  1. event_N/event_data.pt  ← preprocess.py 生成，直接读取
  2. 若不存在，回退到 CSV 解析（慢速模式，同时写入 .pt 缓存）
"""

from typing import Optional, List, Dict
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, HeteroData


class UrbanFloodDataset(InMemoryDataset):
    """Spatio-Temporal Heterogeneous Graph Dataset for Urban Flood Forecasting.

    优先从 static_graph.pt 构建 HeteroData，或从 CSV 回退构建。
    """

    def __init__(self, root: str, model_id: int = 2, split: str = 'train',
                 transform=None, pre_transform=None):
        self.model_id = model_id
        self.split    = split.lower()
        super().__init__(root, transform, pre_transform)
        self._data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False)

    # ── PyG 接口 ─────────────────────────────────────────────────────────────

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root,
                            f'Models/Model_{self.model_id}/{self.split}')

    @property
    def raw_file_names(self) -> List[str]:
        return []   # 禁用 PyG 自动下载；文件由我们自行管理

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return [f'model_{self.model_id}_{self.split}_graph.pt']

    # ── 静态图构建 ────────────────────────────────────────────────────────────

    def process(self):
        """从 static_graph.pt（优先）或 CSV（回退）构建 HeteroData 并缓存。"""
        base       = self.raw_dir
        sg_path    = os.path.join(base, 'static_graph.pt')

        if os.path.exists(sg_path):
            # ── 快速通道：直接从 .pt 文件读取 ────────────────────────────────
            print(f'Loading static graph from {sg_path}')
            sg   = torch.load(sg_path, weights_only=False)
            data = self._build_heterodata_from_sg(sg)
        else:
            # ── 回退通道：从 CSV 读取（并写入 static_graph.pt 缓存）──────────
            print(f'static_graph.pt not found, building from CSV: {base}')
            data = self._build_heterodata_from_csv(base)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self._data, self.slices = self.collate([data])
        os.makedirs(self.processed_dir, exist_ok=True)
        print(f'Saving processed graph to {self.processed_paths[0]}')
        torch.save((self._data, self.slices), self.processed_paths[0])

    # ── 内部：从 static_graph.pt 构建 HeteroData ─────────────────────────────

    @staticmethod
    def _build_heterodata_from_sg(sg: dict) -> HeteroData:
        """将 static_graph.pt 字典转换为 HeteroData 对象。"""
        data = HeteroData()

        # 节点特征 & 原始 ID
        data['manhole'].x_static  = sg['man_static']       # [N1, 4]
        data['manhole'].orig_idx  = sg['man_orig_idx']      # [N1]
        data['cell'].x_static     = sg['cell_static']       # [N2, 6]
        data['cell'].orig_idx     = sg['cell_orig_idx']     # [N2]

        # 边索引
        data['manhole', 'to_manhole', 'manhole'].edge_index = sg['man2man_ei']
        data['cell',    'to_cell',    'cell'   ].edge_index = sg['cell2cell_ei']
        data['manhole', 'to_cell',    'cell'   ].edge_index = sg['man2cell_ei']
        data['cell',    'to_manhole', 'manhole'].edge_index = sg['cell2man_ei']

        # 边静态特征（可选）
        if sg['man2man_attr'].shape[0] > 0:
            data['manhole', 'to_manhole', 'manhole'].edge_attr = sg['man2man_attr']
        if sg['cell2cell_attr'].shape[0] > 0:
            data['cell', 'to_cell', 'cell'].edge_attr = sg['cell2cell_attr']

        return data

    # ── 内部：从 CSV 回退构建 HeteroData（并缓存 static_graph.pt）────────────

    def _build_heterodata_from_csv(self, base: str) -> HeteroData:
        """读取 CSV 文件构建 HeteroData，同时回写 static_graph.pt。"""
        import pandas as _pd
        import numpy as _np

        man_df  = _pd.read_csv(os.path.join(base, '1d_nodes_static.csv'))
        cell_df = _pd.read_csv(os.path.join(base, '2d_nodes_static.csv'))

        man_map  = {int(v): i for i, v in enumerate(man_df['node_idx'].values)}
        cell_map = {int(v): i for i, v in enumerate(cell_df['node_idx'].values)}

        def _f32(df, cols):
            return torch.from_numpy(
                df[cols].fillna(0).astype(_np.float32).values)

        man_static  = _f32(man_df,
            ['depth', 'invert_elevation', 'surface_elevation', 'base_area'])
        cell_static = _f32(cell_df,
            ['area', 'roughness', 'min_elevation', 'elevation', 'aspect', 'curvature'])

        def _ei(df, sm, dm):
            srcs = df['from_node'].astype(int).map(sm).values
            dsts = df['to_node'].astype(int).map(dm).values
            v    = (~_np.isnan(srcs.astype(float))) & (~_np.isnan(dsts.astype(float)))
            return torch.from_numpy(
                _np.vstack([srcs[v], dsts[v]]).astype(_np.int64))

        e1_df  = _pd.read_csv(os.path.join(base, '1d_edge_index.csv'))
        e2_df  = _pd.read_csv(os.path.join(base, '2d_edge_index.csv'))
        cpl_df = _pd.read_csv(os.path.join(base, '1d2d_connections.csv'))

        man2man_ei   = _ei(e1_df, man_map, man_map)
        cell2cell_ei = _ei(e2_df, cell_map, cell_map)

        s1d = cpl_df['node_1d'].astype(int).map(man_map).values
        d2d = cpl_df['node_2d'].astype(int).map(cell_map).values
        vc  = (~_np.isnan(s1d.astype(float))) & (~_np.isnan(d2d.astype(float)))
        s1d = s1d[vc].astype(_np.int64)
        d2d = d2d[vc].astype(_np.int64)

        man2cell_ei = torch.from_numpy(_np.vstack([s1d, d2d]))
        cell2man_ei = torch.from_numpy(_np.vstack([d2d, s1d]))

        def _edge_attr(fname, cols, E):
            p = os.path.join(base, fname)
            if not os.path.exists(p):
                return torch.zeros(E, len(cols), dtype=torch.float32)
            df = _pd.read_csv(p).sort_values('edge_idx').reset_index(drop=True)
            return _f32(df, cols)

        E1 = man2man_ei.shape[1]
        E2 = cell2cell_ei.shape[1]
        man2man_attr   = _edge_attr('1d_edges_static.csv',
            ['relative_position_x', 'relative_position_y',
             'length', 'diameter', 'shape', 'roughness', 'slope'], E1)
        cell2cell_attr = _edge_attr('2d_edges_static.csv',
            ['relative_position_x', 'relative_position_y',
             'face_length', 'length', 'slope'], E2)

        # 写 static_graph.pt 缓存，下次直接读
        sg = {
            'man_static'    : man_static,
            'man_orig_idx'  : torch.from_numpy(man_df['node_idx'].astype(_np.int64).values),
            'cell_static'   : cell_static,
            'cell_orig_idx' : torch.from_numpy(cell_df['node_idx'].astype(_np.int64).values),
            'man2man_ei'    : man2man_ei,
            'man2man_attr'  : man2man_attr,
            'cell2cell_ei'  : cell2cell_ei,
            'cell2cell_attr': cell2cell_attr,
            'man2cell_ei'   : man2cell_ei,
            'cell2man_ei'   : cell2man_ei,
            'N1': len(man_map), 'N2': len(cell_map), 'E1': E1, 'E2': E2,
        }
        torch.save(sg, os.path.join(base, 'static_graph.pt'))
        print(f'  Cached static_graph.pt at {base}')

        return self._build_heterodata_from_sg(sg)

    # ── 动态事件加载 ─────────────────────────────────────────────────────────

    def load_event(self, event_folder: str) -> Dict[str, object]:
        """加载单个事件的动态时序数据。

        返回字典:
          'manhole'   Tensor [T, N1, 2]   water_level, inlet_flow
          'cell'      Tensor [T, N2, 3]   rainfall, water_level, water_volume
          '1d_edges'  Tensor [T, E1, 2]   flow, velocity
          '2d_edges'  Tensor [T, E2, 2]   flow, velocity
          'timesteps' int                  T
          'tstamp_df' DataFrame            timesteps.csv 内容

        优先读取 event_data.pt；不存在时回退到 CSV 并写入缓存。
        """
        base    = self.raw_dir
        ev_path = os.path.join(base, event_folder)
        pt_path = os.path.join(ev_path, 'event_data.pt')

        # ── 快速通道 ──────────────────────────────────────────────────────────
        if os.path.exists(pt_path):
            try:
                d = torch.load(pt_path, weights_only=False)
                # 为旧版 .pt（无边动态）补充边动态字段
                if '1d_edges' not in d or '2d_edges' not in d:
                    d = self._patch_edge_dyn(d, base, ev_path)
                return d
            except Exception as e:
                print(f'⚠️ 读取 {pt_path} 失败，回退 CSV 模式。错误: {e}')

        # ── 慢速通道：CSV 解析 ────────────────────────────────────────────────
        return self._load_event_from_csv(base, ev_path, pt_path)

    # ── 内部：补充旧版 .pt 中缺失的边动态 ────────────────────────────────────

    def _patch_edge_dyn(self, d: dict, base: str, ev_path: str) -> dict:
        T  = d['timesteps']
        E1 = len(pd.read_csv(os.path.join(base, '1d_edge_index.csv')))
        E2 = len(pd.read_csv(os.path.join(base, '2d_edge_index.csv')))
        d['1d_edges'] = _load_edge_dyn_csv(
            os.path.join(ev_path, '1d_edges_dynamic_all.csv'), T, E1)
        d['2d_edges'] = _load_edge_dyn_csv(
            os.path.join(ev_path, '2d_edges_dynamic_all.csv'), T, E2)
        return d

    # ── 内部：从 CSV 加载并缓存 ───────────────────────────────────────────────

    def _load_event_from_csv(self, base: str, ev_path: str,
                             pt_path: str) -> Dict[str, object]:
        """CSV 解析，写 .pt 缓存，与 preprocess.py 逻辑完全一致。"""
        man_df  = pd.read_csv(os.path.join(base, '1d_nodes_static.csv'))
        cell_df = pd.read_csv(os.path.join(base, '2d_nodes_static.csv'))
        man_map  = {int(v): i for i, v in enumerate(man_df['node_idx'].values)}
        cell_map = {int(v): i for i, v in enumerate(cell_df['node_idx'].values)}

        E1 = len(pd.read_csv(os.path.join(base, '1d_edge_index.csv')))
        E2 = len(pd.read_csv(os.path.join(base, '2d_edge_index.csv')))

        man_dyn_fp  = os.path.join(ev_path, '1d_nodes_dynamic_all.csv')
        cell_dyn_fp = os.path.join(ev_path, '2d_nodes_dynamic_all.csv')
        ts_fp       = os.path.join(ev_path, 'timesteps.csv')

        if not os.path.exists(man_dyn_fp):
            raise FileNotFoundError(f'Dynamic file missing: {man_dyn_fp}')

        ts_df = pd.read_csv(ts_fp)
        T     = len(ts_df)
        N1    = len(man_map)
        N2    = len(cell_map)

        man_arr = np.zeros((T, N1, 2), dtype=np.float32)
        for t_idx, grp in pd.read_csv(man_dyn_fp).groupby('timestep'):
            idxs  = grp['node_idx'].map(man_map).values
            valid = ~np.isnan(idxs.astype(float))
            man_arr[int(t_idx), idxs[valid].astype(int), :] = \
                grp[['water_level', 'inlet_flow']].values[valid].astype(np.float32)

        cell_arr = np.zeros((T, N2, 3), dtype=np.float32)
        for t_idx, grp in pd.read_csv(cell_dyn_fp).groupby('timestep'):
            idxs  = grp['node_idx'].map(cell_map).values
            valid = ~np.isnan(idxs.astype(float))
            cell_arr[int(t_idx), idxs[valid].astype(int), :] = \
                grp[['rainfall', 'water_level', 'water_volume']].values[valid].astype(np.float32)

        result = {
            'manhole'   : torch.from_numpy(man_arr),
            'cell'      : torch.from_numpy(cell_arr),
            '1d_edges'  : _load_edge_dyn_csv(
                os.path.join(ev_path, '1d_edges_dynamic_all.csv'), T, E1),
            '2d_edges'  : _load_edge_dyn_csv(
                os.path.join(ev_path, '2d_edges_dynamic_all.csv'), T, E2),
            'timesteps' : T,
            'tstamp_df' : ts_df,
        }

        # 写缓存，下次快速读取
        try:
            torch.save(result, pt_path)
        except Exception as e:
            print(f'⚠️ 写入缓存 {pt_path} 失败: {e}')

        return result


# ─────────────────────── 模块级辅助函数 ──────────────────────────────────────

def _load_edge_dyn_csv(csv_path: str, T: int, E: int) -> torch.Tensor:
    """加载边动态 CSV → Tensor [T, E, 2]（flow, velocity）。"""
    arr = np.zeros((T, E, 2), dtype=np.float32)
    if not os.path.exists(csv_path):
        return torch.from_numpy(arr)
    df = pd.read_csv(csv_path)
    for t_idx, grp in df.groupby('timestep'):
        idxs  = grp['edge_idx'].values.astype(int)
        valid = (idxs >= 0) & (idxs < E)
        arr[int(t_idx), idxs[valid], :] = \
            grp[['flow', 'velocity']].values[valid].astype(np.float32)
    return torch.from_numpy(arr)
