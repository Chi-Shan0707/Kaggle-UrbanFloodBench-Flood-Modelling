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
    
    def forward(self, data: HeteroData, 
                manhole_dyn: torch.Tensor, cell_dyn: torch.Tensor,
                h_prev_dict: Dict[str, torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass for one timestep prediction."""
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
        
        # ========== Step 3: Decode ==========
        pred_manhole_norm = self.manhole_decoder(h_dict['manhole'])  # [N_1d, D1]
        pred_cell_norm = self.cell_decoder(h_dict['cell'])  # [N_2d, D2]

        # ==================== 【输出反归一化 - Multivariate】 ====================
        # Denormalize all features: Y_real = Y_norm * Std + Mean
        # Broadcasting: pred_norm [N, D] * std [D] + mean [D] -> [N, D]
        pred_manhole_real = pred_manhole_norm * self.man_dyn_std + self.man_dyn_mean
        pred_cell_real = pred_cell_norm * self.cell_dyn_std + self.cell_dyn_mean
        # ==================== 【反归一化 END】 ====================

        pred_dict = {'manhole': pred_manhole_real, 'cell': pred_cell_real}
        
        return pred_dict, h_dict
    
    def predict_sequence(self, data: HeteroData, 
                        manhole_dyn_seq: torch.Tensor, 
                        cell_dyn_seq: torch.Tensor,
                        horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively predict a sequence."""
        T_context = manhole_dyn_seq.shape[0]
        
        # Process context
        h_dict = None
        for t in range(T_context):
            _, h_dict = self.forward(data, manhole_dyn_seq[t], cell_dyn_seq[t], h_dict)
        
        # Predict horizon steps
        manhole_preds = []
        cell_preds = []
        
        manhole_dyn_t = manhole_dyn_seq[-1]
        cell_dyn_t = cell_dyn_seq[-1]
        
        for _ in range(horizon):
            pred_dict, h_dict = self.forward(data, manhole_dyn_t, cell_dyn_t, h_dict)
            manhole_preds.append(pred_dict['manhole'])
            cell_preds.append(pred_dict['cell'])
            
            # Update dynamic features for autoregression
            manhole_dyn_t = manhole_dyn_t.clone()
           
            manhole_dyn_t[:, 0] = pred_dict['manhole'][:, 0] # 明确取第0列
            cell_dyn_t = cell_dyn_t.clone()
            cell_dyn_t[:, 1] = pred_dict['cell'][:, 1]
        
        return torch.stack(manhole_preds), torch.stack(cell_preds)