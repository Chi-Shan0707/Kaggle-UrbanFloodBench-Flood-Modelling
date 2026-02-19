"""Configuration for UrbanFloodBench GNN Training Pipeline."""

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """Hyperparameters for HeteroFloodGNN."""
    
    # Architecture
    hidden_dim: int = 128  # Hidden dimension D for all node types
    num_gnn_layers: int = 3  # Number of message passing layers
    num_recurrent_steps: int = 3  # Number of GRU-GNN steps per prediction
    
    # GNN Layer Type
    use_gatv2: bool = True  # If True use GATv2Conv, else use GENConv
    num_heads: int = 4  # For GATv2Conv
    dropout: float = 0.1
    
    # Encoder/Decoder
    encoder_layers: List[int] = None  # MLP layers, None defaults to [hidden_dim]
    decoder_layers: List[int] = None  # MLP layers, None defaults to [hidden_dim//2]
    
    def __post_init__(self):
        if self.encoder_layers is None:
            self.encoder_layers = [self.hidden_dim]
        if self.decoder_layers is None:
            self.decoder_layers = [self.hidden_dim // 2]


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    
    # Data
    model_id: int = 1
    train_events: List[str] = None  # e.g. ['event_1', 'event_2', ...]
    val_events: List[str] = None
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1  # Events are large, usually process one at a time
    num_epochs: int = 10 # 原50
    gradient_clip: float = 1.0
    
    # Autoregressive Training
    teacher_forcing_ratio_start: float = 1.0  # Start with 100% ground truth
    teacher_forcing_ratio_end: float = 0.2  # End with 20% ground truth
    teacher_forcing_decay_epochs: int = 30  # Decay over N epochs
    
    # Loss
    use_standardized_rmse: bool = True
    # Standard deviations for normalization (computed from training data)
    std_manhole: float = 1.0  # Placeholder, compute from data
    std_cell: float = 1.0  # Placeholder, compute from data
    
    # Logging & Checkpointing
    log_every_n_steps: int = 10
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    def __post_init__(self):
    # 采用硬编码，分配event
        # Model_1: 69 events (80% train, 20% validate)
        # Model_2: 72 events (80% train, 20% validate)
        model1_all_events = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 30, 32, 34, 36, 38, 39, 40, 41, 43, 45, 46, 47, 49, 50, 54, 55, 56, 57, 58, 60, 61, 63, 64, 68, 70, 71, 72, 74, 76, 77, 78, 79, 82, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96]
        model2_all_events = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30, 32, 33, 34, 36, 38, 39, 40, 41, 43, 45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 63, 64, 68, 69, 70, 71, 72, 74, 75, 78, 79, 80, 81, 83, 85, 86, 87, 89, 91, 92, 93, 94, 95, 96, 97, 98]
        
        if self.train_events is None or self.val_events is None:
            if self.model_id == 1:
                # Model_1: 69 events → ~55 train, ~14 validate
                split_idx = int(len(model1_all_events) * 0.8)
                train_list = model1_all_events[:split_idx]
                val_list = model1_all_events[split_idx:]
            else:  # model_id == 2
                # Model_2: 72 events → ~57 train, ~15 validate
                split_idx = int(len(model2_all_events) * 0.8)
                train_list = model2_all_events[:split_idx]
                val_list = model2_all_events[split_idx:]
            
            if self.train_events is None:
                self.train_events = [f'event_{i}' for i in train_list]
            if self.val_events is None:
                self.val_events = [f'event_{i}' for i in val_list]
