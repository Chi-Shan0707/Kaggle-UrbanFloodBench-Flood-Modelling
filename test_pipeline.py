"""Quick test script to verify the pipeline is working correctly."""

import torch
import numpy as np
from pathlib import Path

def test_dataset():
    """Test dataset loading."""
    print("=" * 60)
    print("Testing Dataset...")
    print("=" * 60)
    
    from dataset import UrbanFloodDataset
    
    try:
        dataset = UrbanFloodDataset(root="./", model_id=2)
        data = dataset.get(0)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Manholes: {data['manhole'].x_static.shape}")
        print(f"  - Cells: {data['cell'].x_static.shape}")
        print(f"  - 1D edges: {data['manhole', 'to_manhole', 'manhole'].edge_index.shape}")
        print(f"  - 2D edges: {data['cell', 'to_cell', 'cell'].edge_index.shape}")
        print(f"  - 1D→2D coupling: {data['manhole', 'to_cell', 'cell'].edge_index.shape}")
        print(f"  - 2D→1D coupling: {data['cell', 'to_manhole', 'manhole'].edge_index.shape}")
        
        # 【修复点 1】 使用 dataset.load_event 而不是 data.load_event
        event_data = dataset.load_event('event_1')
        print(f"\n✓ Event loaded successfully")
        print(f"  - Manhole dynamics: {event_data['manhole'].shape}")
        print(f"  - Cell dynamics: {event_data['cell'].shape}")
        print(f"  - Timesteps: {event_data['timesteps']}")
        
        return True, data
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model(data):
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model...")
    print("=" * 60)
    
    from model import HeteroFloodGNN
    from config import ModelConfig
    from dataset import UrbanFloodDataset
    
    try:
        config = ModelConfig(
            hidden_dim=64,  # Small for testing
            num_gnn_layers=2,
            num_recurrent_steps=2,
            use_gatv2=True,
            num_heads=2
        )
        
        model = HeteroFloodGNN(
            config=config,
            manhole_static_dim=4,
            cell_static_dim=6,
            manhole_dynamic_dim=2,
            cell_dynamic_dim=3
        )
        
        print(f"✓ Model initialized")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        # 【修复点 2】 需要 dataset 实例来加载事件数据
        dataset = UrbanFloodDataset(root="./", model_id=2)
        event_data = dataset.load_event('event_1')
        
        manhole_dyn = event_data['manhole'][0]  # First timestep
        cell_dyn = event_data['cell'][0]
        
        # forward 返回 3-tuple: (pred_dict, h_dict, cell_to_cell_flow)
        pred_dict, h_dict, cell_flow = model(data, manhole_dyn, cell_dyn)
        
        print(f"\n✓ Forward pass successful")
        print(f"  - Manhole predictions: {pred_dict['manhole'].shape}")
        print(f"  - Cell predictions: {pred_dict['cell'].shape}")
        print(f"  - Manhole hidden state: {h_dict['manhole'].shape}")
        print(f"  - Cell hidden state: {h_dict['cell'].shape}")
        print(f"  - Cell→Cell edge flow: {cell_flow.shape}  (隐式边流量预测)")
        
        # Test sequence prediction
        print(f"\n✓ Testing sequence prediction...")
        manhole_seq = event_data['manhole'][:5]  # First 5 timesteps
        cell_seq = event_data['cell'][:5]
        
        manhole_preds, cell_preds = model.predict_sequence(
            data, manhole_seq, cell_seq, horizon=3
        )
        
        print(f"  - Predicted {manhole_preds.shape[0]} future timesteps")
        print(f"  - Manhole predictions: {manhole_preds.shape}")
        print(f"  - Cell predictions: {cell_preds.shape}")
        
        return True, model
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_loss():
    """Test loss function."""
    print("\n" + "=" * 60)
    print("Testing Loss Function...")
    print("=" * 60)
    
    from train import StandardizedRMSELoss
    
    try:
        criterion = StandardizedRMSELoss(std_manhole=2.0, std_cell=1.5)
        
        # Dummy predictions and targets
        pred_dict = {
            'manhole': torch.randn(121, 1),
            'cell': torch.randn(5213, 1)
        }
        target_dict = {
            'manhole': torch.randn(121, 1),
            'cell': torch.randn(5213, 1)
        }
        
        loss = criterion(pred_dict, target_dict)
        
        print(f"✓ Loss computation successful")
        print(f"  - Loss value: {loss.item():.4f}")
        
        # Test with masks
        mask_dict = {
            'manhole': torch.rand(121) > 0.5,
            'cell': torch.rand(5213) > 0.5
        }
        
        loss_masked = criterion(pred_dict, target_dict, mask_dict)
        print(f"  - Masked loss value: {loss_masked.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(data, model):
    """Test one training step."""
    print("\n" + "=" * 60)
    print("Testing Training Step...")
    print("=" * 60)
    
    from train import train_one_event, StandardizedRMSELoss
    from dataset import UrbanFloodDataset
    import torch.optim as optim
    
    try:
        criterion = StandardizedRMSELoss(std_manhole=2.0, std_cell=1.5)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 【修复点 3】 使用 dataset.load_event
        dataset = UrbanFloodDataset(root="./", model_id=2)
        event_data = dataset.load_event('event_1')
        
        print(f"  Running training step on event_1...")
        optimizer.zero_grad()
        
        # Use only first 10 timesteps for speed
        event_data_short = {
            'manhole': event_data['manhole'][:10],
            'cell': event_data['cell'][:10],
            'timesteps': 10
        }
        
        from train import train_one_event
        loss = train_one_event(
            model, data, event_data_short, criterion, 
            device='cpu', teacher_forcing_ratio=1.0
        )
        
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  - Loss: {loss:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("UrbanFloodBench GNN Pipeline - Test Suite")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test 1: Dataset
    success, data = test_dataset()
    results['Dataset'] = success
    
    if not success:
        print("\n⚠️ Cannot continue without dataset. Please check data paths.")
        return
    
    # Test 2: Model
    success, model = test_model(data)
    results['Model'] = success
    
    # Test 3: Loss
    success = test_loss()
    results['Loss'] = success
    
    # Test 4: Training Step
    if model is not None:
        success = test_training_step(data, model)
        results['Training'] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Configure hyperparameters in config.py")
        print("  2. Run: python train.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()