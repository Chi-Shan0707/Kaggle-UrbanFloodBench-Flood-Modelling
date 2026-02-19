"""HeteroFloodGNN: Spatio-Temporal Heterogeneous GNN for Urban Flood Forecasting.

Architecture:
    Encoder → Recurrent-Processor → Decoder

The processor uses a GRU-style recurrent mechanism where the linear operations
are replaced by heterogeneous graph convolutions (HeteroConv).

Tensor Shapes:
    - manhole features: [N_1d, D]
    - cell features: [N_2d, D]
    - hidden states h_t: same as features
    - output: [N_1d, 1] for manholes, [N_2d, 1] for cells (water level predictions)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, GENConv, Linear

from config import ModelConfig


class HeteroFloodGNN(nn.Module):
    """Heterogeneous GNN with Recurrent Message Passing for flood prediction.
    
    The model processes static and dynamic features through:
    1. Node-type-specific encoders (MLP)
    2. Recurrent GNN processor (GRU-GNN hybrid)
    3. Node-type-specific decoders (MLP) → scalar water level prediction
    """
    
    def __init__(self, config: ModelConfig, 
                 manhole_static_dim: int, cell_static_dim: int,
                 manhole_dynamic_dim: int, cell_dynamic_dim: int):
        """
        Args:
            config: Model hyperparameters
            manhole_static_dim: # of static features for manholes (e.g., 4)
            cell_static_dim: # of static features for cells (e.g., 6)
            manhole_dynamic_dim: # of dynamic features per timestep for manholes (e.g., 2)
            cell_dynamic_dim: # of dynamic features per timestep for cells (e.g., 3)
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_recurrent_steps = config.num_recurrent_steps

        # ==================== 【新增代码 START】 ====================
        # 注册归一化参数 (Buffer 会随模型保存，但不会被梯度更新)
        # 初始化为 0均值 1标准差 (防止未注入参数时报错)
        self.register_buffer('man_dyn_mean', torch.zeros(manhole_dynamic_dim))
        self.register_buffer('man_dyn_std',  torch.ones(manhole_dynamic_dim))
        self.register_buffer('cell_dyn_mean', torch.zeros(cell_dynamic_dim))
        self.register_buffer('cell_dyn_std',  torch.ones(cell_dynamic_dim))
        # ==================== 【新增代码 END】 ====================

        # Input dimensions: concatenate static + dynamic features
        manhole_input_dim = manhole_static_dim + manhole_dynamic_dim
        cell_input_dim = cell_static_dim + cell_dynamic_dim
        
        # ========== Encoder: Project input features to hidden_dim ==========
        self.manhole_encoder = self._build_mlp(manhole_input_dim, config.hidden_dim, config.encoder_layers)
        self.cell_encoder = self._build_mlp(cell_input_dim, config.hidden_dim, config.encoder_layers)
        
        # ========== Recurrent Processor: GRU-GNN hybrid ==========
        # GRU Logic: Input to gates is cat([x, h]), so input dim is 2 * hidden_dim
        gru_input_dim = 2 * config.hidden_dim
        
        self.reset_conv = self._build_hetero_conv(config, gru_input_dim, config.hidden_dim)
        self.update_conv = self._build_hetero_conv(config, gru_input_dim, config.hidden_dim)
        self.candidate_conv = self._build_hetero_conv(config, gru_input_dim, config.hidden_dim)
        
        # ========== Decoder: hidden_dim → all dynamic features ==========
        self.manhole_decoder = self._build_mlp(config.hidden_dim, manhole_dynamic_dim, config.decoder_layers)
        self.cell_decoder = self._build_mlp(config.hidden_dim, cell_dynamic_dim, config.decoder_layers)

        # ========== 隐式边流量解码器 (Edge Flow Decoder) ==========
        # 【为物理质量守恒损失做准备】
        # 将边两端节点的隐状态拼接（2 * hidden_dim）作为输入，预测该边上的标量流量值 Q。
        # 数据来源: *_edges_dynamic_all.csv 中的 'flow' 列可作为训练时监督信号。
        # 输入维度: 2 * hidden_dim  → cat([h_src, h_dst])，形状 [E, 2*D]
        # 输出维度: 1               → 标量流量预测值，形状 [E, 1]
        self.edge_decoder = self._build_mlp(2 * config.hidden_dim, 1, config.decoder_layers)

        # 注册边流量归一化参数（Buffer 随模型保存，但不参与梯度更新）
        # 未注入真实统计值时默认为恒等变换 (mean=0, std=1)
        self.register_buffer('edge_flow_mean', torch.zeros(1))
        self.register_buffer('edge_flow_std',  torch.ones(1))

    def _build_mlp(self, in_dim: int, out_dim: int, hidden_layers: list) -> nn.Sequential:
        """Build a simple MLP with ReLU activations."""
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)
    
    def _build_hetero_conv(self, config: ModelConfig, in_dim: int, out_dim: int) -> HeteroConv:
        """Build a HeteroConv layer with specified dimensions.
        
        Args:
            config: Model configuration
            in_dim: Input dimension (e.g. 2*D for GRU gates)
            out_dim: Output dimension (e.g. D)
        """
        if config.use_gatv2:
            # GATv2Conv output size is heads * out_channels. 
            # We want total output to be out_dim, so per-head dim is out_dim // heads
            per_head_dim = out_dim // config.num_heads
            
            conv_dict = {
                # Case 1: Homo-edges (Self-loops OK)
                ('manhole', 'to_manhole', 'manhole'): GATv2Conv(in_dim, per_head_dim, 
                                                                  heads=config.num_heads, 
                                                                  concat=True, dropout=config.dropout),
                ('cell', 'to_cell', 'cell'): GATv2Conv(in_dim, per_head_dim, 
                                                         heads=config.num_heads, 
                                                         concat=True, dropout=config.dropout),
                
                # Case 2: Bipartite edges (No Self-loops)
                ('manhole', 'to_cell', 'cell'): GATv2Conv(in_dim, per_head_dim, 
                                                            heads=config.num_heads, 
                                                            concat=True, dropout=config.dropout,
                                                            add_self_loops=False),
                
                ('cell', 'to_manhole', 'manhole'): GATv2Conv(in_dim, per_head_dim, 
                                                               heads=config.num_heads, 
                                                               concat=True, dropout=config.dropout,
                                                               add_self_loops=False),
            }
        else:
            # GENConv
            conv_dict = {
                ('manhole', 'to_manhole', 'manhole'): GENConv(in_dim, out_dim, aggr='softmax', norm='layer'),
                ('cell', 'to_cell', 'cell'): GENConv(in_dim, out_dim, aggr='softmax', norm='layer'),
                ('manhole', 'to_cell', 'cell'): GENConv(in_dim, out_dim, aggr='softmax', norm='layer'),
                ('cell', 'to_manhole', 'manhole'): GENConv(in_dim, out_dim, aggr='softmax', norm='layer'),
            }
        
        return HeteroConv(conv_dict, aggr='sum')
    
    def _gru_step(self, x_dict: Dict[str, torch.Tensor], 
                  h_dict: Dict[str, torch.Tensor],
                  edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """One step of GRU-GNN."""
        # Concatenate x and h -> Shape becomes [N, 2*D]
        xh_dict = {k: torch.cat([x_dict[k], h_dict[k]], dim=-1) for k in x_dict.keys()}
        
        # Reset gate: r_t = sigmoid(Conv([x_t, h_{t-1}]))
        r_dict_raw = self.reset_conv(xh_dict, edge_index_dict)
        r_dict = {k: torch.sigmoid(v) for k, v in r_dict_raw.items()}
        
        # Update gate: z_t = sigmoid(Conv([x_t, h_{t-1}]))
        z_dict_raw = self.update_conv(xh_dict, edge_index_dict)
        z_dict = {k: torch.sigmoid(v) for k, v in z_dict_raw.items()}
        
        # Candidate: h_tilde = tanh(Conv([x_t, r_t * h_{t-1}]))
        # Note: r_t * h_{t-1} creates a tensor of size D. 
        # We concat x (size D) with it, so input is again size 2*D.
        rh_dict = {k: torch.cat([x_dict[k], r_dict[k] * h_dict[k]], dim=-1) for k in x_dict.keys()}
        h_tilde_dict_raw = self.candidate_conv(rh_dict, edge_index_dict)
        h_tilde_dict = {k: torch.tanh(v) for k, v in h_tilde_dict_raw.items()}
        
        # Update: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        h_new_dict = {k: (1 - z_dict[k]) * h_dict[k] + z_dict[k] * h_tilde_dict[k] 
                      for k in h_dict.keys()}
        
        return h_new_dict

    def predict_edge_flow(self,
                          h_nodes: torch.Tensor,
                          edge_index: torch.Tensor) -> torch.Tensor:
        """预测边上的标量流量值（隐式边流量预测）。

        【隐式边流量预测 - 为物理质量守恒损失做准备】
        将边的源节点与目标节点的隐状态拼接后，经过 edge_decoder 预测该边上的
        流量 Q（标量）。这是一个"隐变量"预测：
          - 训练时：可用真实边流量数据（来自 *_edges_dynamic_all.csv 的 'flow' 列）
                    作为辅助监督信号，使模型感知管道/网格间的水流传输规律。
          - 推理时：可用于构建物理质量守恒约束 ∑Q_in = ∑Q_out + ΔV/Δt，
                    对节点预测值进行物理后处理修正。

        Args:
            h_nodes:    节点隐状态，形状 [N, D]
                        （源节点与目标节点属于同一类型时共用同一张量）
            edge_index: 有向边索引，形状 [2, E]
                          edge_index[0] → 源节点全局索引 [E]
                          edge_index[1] → 目标节点全局索引 [E]

        Returns:
            edge_flow:  边流量预测值，形状 [E, 1]
        """
        src_idx = edge_index[0]                           # [E] 源节点索引
        dst_idx = edge_index[1]                           # [E] 目标节点索引

        h_src = h_nodes[src_idx]                          # [E, D] 源节点隐状态
        h_dst = h_nodes[dst_idx]                          # [E, D] 目标节点隐状态

        # 拼接源与目标节点隐状态 → 形状 [E, 2*D]
        # 这同时编码了流量的"方向性"（从 src 到 dst）
        edge_input = torch.cat([h_src, h_dst], dim=-1)    # [E, 2*D]

        # 通过解码器预测标量流量 → 形状 [E, 1]
        edge_flow = self.edge_decoder(edge_input)          # [E, 1]

        return edge_flow

    def forward(self, data: HeteroData, 
                manhole_dyn: torch.Tensor, cell_dyn: torch.Tensor,
                h_prev_dict: Dict[str, torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for one timestep prediction.

        Returns:
            pred_dict:          节点预测字典 {'manhole': [N1, D1], 'cell': [N2, D2]}
            h_dict:             更新后的隐状态字典 {'manhole': [N1, D], 'cell': [N2, D]}
            cell_to_cell_flow:  2D cell→cell 边流量预测 [E_2d, 1]，可用于质量守恒损失
        """
        device = manhole_dyn.device

        # ==================== 【新增代码 START: 输入归一化】 ====================
        # (X - Mean) / (Std + eps)
        # 加上 1e-6 是为了防止除以 0
        manhole_dyn_norm = (manhole_dyn - self.man_dyn_mean) / (self.man_dyn_std + 1e-6)
        cell_dyn_norm = (cell_dyn - self.cell_dyn_mean) / (self.cell_dyn_std + 1e-6)
        # ==================== 【新增代码 END】 ====================

        # ========== Step 1: Encode ==========
        manhole_feats = torch.cat([data['manhole'].x_static.to(device), manhole_dyn_norm], dim=-1)
        cell_feats = torch.cat([data['cell'].x_static.to(device), cell_dyn_norm], dim=-1)
        
        x_manhole = self.manhole_encoder(manhole_feats)  # [N_1d, D]
        x_cell = self.cell_encoder(cell_feats)  # [N_2d, D]
        
        x_dict = {'manhole': x_manhole, 'cell': x_cell}
        
        # Initialize hidden state if not provided
        if h_prev_dict is None:
            h_prev_dict = {
                'manhole': torch.zeros_like(x_manhole),
                'cell': torch.zeros_like(x_cell)
            }
        
        # ========== Step 2: Recurrent Message Passing ==========
        h_dict = h_prev_dict
        edge_index_dict = data.edge_index_dict
        
        for _ in range(self.num_recurrent_steps):
            h_dict = self._gru_step(x_dict, h_dict, edge_index_dict)
        
        # ========== Step 3: Decode（残差解码）==========
        # 【设计思路 - 参考 DUALFloodGNN】
        # 传统方式：模型直接预测下一时刻的绝对值，自回归时误差会逐步累积放大。
        # 残差方式：模型只需预测归一化空间中的"变化量 (delta)"，绝对值基数由输入提供。
        # 好处：
        #   1. 学习目标从 [0, ∞) 的绝对水位值变为接近 0 的小增量，更容易优化。
        #   2. 自回归时，即使单步预测有偏差，残差叠加的误差比直接预测绝对值更稳定。
        #   3. 梯度信号更干净，不会被静态偏置项主导。

        # 解码器输出的是归一化空间中的增量 Δx_norm，而非绝对值
        # 形状: delta_manhole_norm [N_1d, D1=2], delta_cell_norm [N_2d, D2=3]
        delta_manhole_norm = self.manhole_decoder(h_dict['manhole'])  # [N_1d, D1]，归一化增量
        delta_cell_norm    = self.cell_decoder(h_dict['cell'])         # [N_2d, D2]，归一化增量

        # 【残差连接】: 下一时刻归一化预测值 = 当前时刻归一化输入 + 模型预测的归一化增量
        # 公式: x̂_{t+1,norm} = x_{t,norm} + Δx_norm
        # 张量形状校验:
        #   manhole_dyn_norm: [N_1d, D1]  +  delta_manhole_norm: [N_1d, D1]  → [N_1d, D1] ✓
        #   cell_dyn_norm:    [N_2d, D2]  +  delta_cell_norm:    [N_2d, D2]  → [N_2d, D2] ✓
        pred_manhole_norm = manhole_dyn_norm + delta_manhole_norm  # [N_1d, D1]
        pred_cell_norm    = cell_dyn_norm    + delta_cell_norm      # [N_2d, D2]

        # ==================== 【输出反归一化 - Multivariate】 ====================
        # 将归一化预测值还原到真实物理量空间: Y_real = Y_norm * Std + Mean
        # Broadcasting: pred_norm [N, D] * std [D] + mean [D] -> [N, D]
        pred_manhole_real = pred_manhole_norm * self.man_dyn_std + self.man_dyn_mean
        pred_cell_real    = pred_cell_norm    * self.cell_dyn_std  + self.cell_dyn_mean
        # ==================== 【反归一化 END】 ====================

        # ========== Step 4: 隐式边流量预测（2D cell→cell 边）==========
        # 【隐式边流量预测】
        # 利用经过全部 GRU-GNN 步骤更新后的 cell 节点隐状态，
        # 预测 2D 网格中每条 cell→cell 边上的流量值。
        # 这一步在训练和推理中均会执行：
        #   - 训练时：可与 2d_edges_dynamic_all.csv 中的真实 flow 对比，
        #             构建辅助监督损失（物理质量守恒约束）。
        #   - 推理时：作为隐变量输出，供下游物理后处理使用。
        # 张量形状: cell_to_cell_flow [E_2d, 1]
        #   其中 E_2d = data['cell', 'to_cell', 'cell'].edge_index.shape[1]
        cell_edge_index = data['cell', 'to_cell', 'cell'].edge_index  # [2, E_2d]
        cell_to_cell_flow = self.predict_edge_flow(
            h_dict['cell'], cell_edge_index
        )  # [E_2d, 1]

        pred_dict = {'manhole': pred_manhole_real, 'cell': pred_cell_real}

        # 返回三元组: (节点预测字典, 隐状态字典, 2D 边流量预测)
        # cell_to_cell_flow 可供训练脚本构建物理质量守恒辅助损失
        return pred_dict, h_dict, cell_to_cell_flow

    def predict_sequence(self, data: HeteroData, 
                        manhole_dyn_seq: torch.Tensor, 
                        cell_dyn_seq: torch.Tensor,
                        horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively predict a sequence."""
        T_context = manhole_dyn_seq.shape[0]
        
        # Process context
        h_dict = None
        for t in range(T_context):
            # forward 返回 3-tuple；此处只需更新隐状态
            _, h_dict, _ = self.forward(data, manhole_dyn_seq[t], cell_dyn_seq[t], h_dict)
        
        # Predict horizon steps
        manhole_preds = []
        cell_preds = []
        
        manhole_dyn_t = manhole_dyn_seq[-1]
        cell_dyn_t = cell_dyn_seq[-1]
        
        for _ in range(horizon):
            # forward 返回 3-tuple；丢弃边流量预测（此处不用于序列生成）
            pred_dict, h_dict, _ = self.forward(data, manhole_dyn_t, cell_dyn_t, h_dict)
            manhole_preds.append(pred_dict['manhole'])
            cell_preds.append(pred_dict['cell'])
            
            # Update dynamic features for autoregression (use full predictions)
            manhole_dyn_t = pred_dict['manhole'].clone()
            cell_dyn_t = pred_dict['cell'].clone()
            
            # Note: For true forecasting, rainfall should come from weather predictions
            # This method assumes rainfall continues from last context value
        
        return torch.stack(manhole_preds), torch.stack(cell_preds)