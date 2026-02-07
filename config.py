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
    model_id: int = 2
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
        if self.train_events is None:
            # Default: use first 60 events for training
            self.train_events = [f'event_{i}' for i in [1, 2, 3, 5, 6, 7, 9, 10]]
        if self.val_events is None:
            # Default: use next events for validation
            self.val_events = [f'event_{i}' for i in [11, 12, 13]]
