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


def train_one_event(model: HeteroFloodGNN, data, event_data: Dict, 
                   criterion: nn.Module, device: str,
                   teacher_forcing_ratio: float = 1.0) -> float:
    """Train on one event with autoregressive teacher forcing.
    
    Args:
        model: HeteroFloodGNN model
        data: HeteroData with static features
        event_data: Dict with 'manhole' [T, N1, D1], 'cell' [T, N2, D2]
        criterion: Loss function
        device: torch device
        teacher_forcing_ratio: Probability of using ground truth
    
    Returns:
        Average loss for this event
    """
    model.train()
    
    manhole_seq = event_data['manhole'].to(device)  # [T, N1, D1]
    cell_seq = event_data['cell'].to(device)  # [T, N2, D2]
    T = manhole_seq.shape[0]
    
    total_loss = 0.0
    h_dict = None
    
    # Autoregressive loop: predict t+1 given 0..t
    for t in range(T - 1):
        # Current timestep features
        manhole_dyn_t = manhole_seq[t]  # [N1, D1]
        cell_dyn_t = cell_seq[t]  # [N2, D2]
        
        # Predict next timestep
        pred_dict, h_dict = model(data, manhole_dyn_t, cell_dyn_t, h_dict)
        
        # Ground truth for t+1 (all features)
        target_dict = {
            'manhole': manhole_seq[t+1],  # [N1, D1] - all features
            'cell': cell_seq[t+1]  # [N2, D2] - all features
        }
        
        # Compute loss
        loss = criterion(pred_dict, target_dict)
        total_loss += loss.item()
        
        # Backprop per step
        loss.backward()
        
        # 【关键修复】Detach hidden state to prevent backpropagating through time indefinitely
        # This cuts the graph connection to the previous timestep
        h_dict = {k: v.detach() for k, v in h_dict.items()}
        
        # Teacher forcing: decide whether to use ground truth or prediction for next step
        with torch.no_grad():
            use_teacher = np.random.rand() < teacher_forcing_ratio
            
            if use_teacher:
                # Use ground truth for next iteration
                pass  # manhole_seq[t+1] already has ground truth
            else:
                # Use model prediction for next iteration
                # Inject real rainfall (Cell index 0) to maintain physical consistency
                manhole_seq[t+1] = pred_dict['manhole'].detach()  # [N1, D1]
                
                next_cell = pred_dict['cell'].detach()  # [N2, D2]
                # Preserve ground truth rainfall (index 0)
                true_rainfall = cell_seq[t+1, :, 0].clone()
                next_cell[:, 0] = true_rainfall
                cell_seq[t+1] = next_cell
    
    return total_loss / (T - 1)


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
        pred_dict, h_dict = model(data, manhole_dyn_t, cell_dyn_t, h_dict)
        
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
            
            optimizer.zero_grad()
            loss = train_one_event(model, data, event_data, criterion, device, tf_ratio)
            train_losses.append(loss)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
            optimizer.step()
        
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
