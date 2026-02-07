# Installation Guide for UrbanFloodBench GNN Pipeline

## Prerequisites

- **Python**: 3.11+
- **CUDA**: 13.0 (for RTX 5060 Blackwell)
- **OS**: Linux (tested on Ubuntu/WSL2)

## Step-by-Step Installation

### Option 1: Using pip (Recommended)

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv_flood
source venv_flood/bin/activate  # On Windows: venv_flood\Scripts\activate

# 2. Install PyTorch with CUDA 13.0 support
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 --extra-index-url https://download.pytorch.org/whl/cu130

# 3. Install PyTorch Geometric and dependencies
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.9.1+cu130.html

# 4. Install other dependencies
pip install pandas numpy tqdm matplotlib

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

### Option 2: Using conda

```bash
# 1. Create conda environment
conda create -n flood_gnn python=3.11 -y
conda activate flood_gnn

# 2. Install PyTorch
conda install pytorch==2.9.1 torchvision pytorch-cuda=13.0 -c pytorch -c nvidia -y

# 3. Install PyG
conda install pyg -c pyg -y

# 4. Install dependencies
conda install pandas numpy tqdm matplotlib -y

# 5. Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Option 3: From requirements.txt

```bash
# Install from the project requirements file
pip install -r requirements.txt

# For PyG extensions (torch-scatter, torch-sparse)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.9.1+cu130.html
```

## Verify Installation

Run the test suite:

```bash
cd /mnt/d/CS/Kaggle/UrbanFloodBench-Flood\ Modelling
python test_pipeline.py
```

Expected output:
```
============================================================
UrbanFloodBench GNN Pipeline - Test Suite
============================================================

============================================================
Testing Dataset...
============================================================
✓ Dataset loaded successfully
  - Manholes: torch.Size([121, 4])
  - Cells: torch.Size([5213, 6])
  ...

============================================================
Test Summary
============================================================
Dataset             : ✓ PASS
Model               : ✓ PASS
Loss                : ✓ PASS
Training            : ✓ PASS

============================================================
🎉 All tests passed! Pipeline is ready to use.
============================================================
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install PyTorch first
```bash
pip install torch==2.9.1+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
```

### Issue: `ImportError: cannot import name 'HeteroData'`

**Solution**: Install PyTorch Geometric
```bash
pip install torch-geometric
```

### Issue: `OSError: ... torch_scatter ... not found`

**Solution**: Install with wheel link
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.9.1+cu130.html
```

### Issue: CUDA not available

**Check CUDA installation**:
```bash
nvidia-smi
nvcc --version
```

**Ensure PyTorch CUDA version matches**:
```python
import torch
print(torch.version.cuda)  # Should be 13.0 or compatible
```

### Issue: Out of Memory during training

**Solutions**:
1. Reduce batch size (already 1 in default config)
2. Reduce `hidden_dim` in `config.py` (e.g., 128 → 64)
3. Reduce `num_recurrent_steps` (e.g., 3 → 2)
4. Process fewer timesteps per event during training

## Quick Validation

After installation, run this minimal test:

```python
import torch
import torch_geometric
from torch_geometric.data import HeteroData

print(f"✓ PyTorch {torch.__version__}")
print(f"✓ PyG {torch_geometric.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# Test HeteroData creation
data = HeteroData()
data['node_type'].x = torch.randn(10, 5)
print(f"✓ HeteroData works")

print("\n🎉 All core dependencies installed correctly!")
```

## Next Steps

Once installation is complete:

1. **Test the pipeline**: `python test_pipeline.py`
2. **Configure hyperparameters**: Edit `config.py`
3. **Start training**: `python train.py`
4. **Monitor progress**: Check `./checkpoints/`
5. **Generate submission**: `python inference.py --checkpoint ./checkpoints/best_model.pt`

## Hardware Requirements

**Minimum**:
- GPU: 8GB VRAM (e.g., RTX 3070)
- RAM: 16GB
- Storage: 10GB for data + models

**Recommended** (Your setup):
- GPU: RTX 5060 (16GB VRAM) ✓
- RAM: 32GB+
- Storage: 50GB SSD

## Additional Resources

- PyTorch: https://pytorch.org/get-started/locally/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Competition: https://www.kaggle.com/competitions/urbanfloodbench

## Support

If you encounter issues:
1. Check PyTorch/PyG versions match CUDA version
2. Verify data paths in `dataset.py`
3. Run `test_pipeline.py` for diagnostic info
4. Check GPU memory with `nvidia-smi`
