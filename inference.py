"""Inference script for generating Kaggle submission file.

Features:
1. Loads a specific checkpoint for a specific Model ID.
2. Fixes 'weights_only' error during torch.load.
3. Intelligent Merge: Appends to existing submission file if it exists.
4. **Memory Optimization**: Writes results to CSV incrementally (event by event) 
   to avoid 'Killed' (OOM) errors.
"""
"""
💡 核心改进点
逐行写入 (Append Write)：每处理完 1 个 Event 的所有时间步，直接把数据转化为字符串写入硬盘，然后从内存中删除。
这保证了内存占用只与单个 Event 的大小有关，而与 Event 总数无关。

显式垃圾回收 (gc.collect)：每轮循环后强制清理 Python 垃圾对象和 PyTorch 显存缓存。

字符串拼接优化：直接构建 CSV 格式的字符串列表 f"{...}"，比维护一个巨大的 Pandas DataFrame 更省内存。

注意： 生成的文件中 row_id 暂时都是 -1。这是为了流式写入的效率。你需要最后单独运行一个简单的脚本（或者用 Pandas 打开再保存一次）来重置 row_id。
如果你的内存足够加载最终的 CSV（通常几十 MB 到几百 MB），我在代码末尾留了一个 post_process_submission 函数，你可以取消注释或者单独调用它。
"""

import os
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import gc  # Garbage Collector

from dataset import UrbanFloodDataset
from model import HeteroFloodGNN
from config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference_stream(checkpoint_path: str, model_id: int, output_csv: str, device):
    """Run inference and write to CSV incrementally to save memory."""
    
    # 1. Load Model
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise e

    model_config = checkpoint['model_config']
    
    model = HeteroFloodGNN(
        config=model_config,
        manhole_static_dim=4,
        cell_static_dim=6,
        manhole_dynamic_dim=2,
        cell_dynamic_dim=3
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Model {model_id} loaded. Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # 2. Setup Dataset
    test_dir_path = Path(f"./Models/Model_{model_id}/test")
    split_mode = 'test' if test_dir_path.exists() else 'train'
    
    if split_mode == 'train':
        logger.warning(f"Test directory not found. Falling back to TRAIN data!")

    dataset = UrbanFloodDataset(root="./", model_id=model_id, split=split_mode)
    
    if not os.path.exists(dataset.processed_paths[0]):
        logger.info(f"Processed graph for {split_mode} not found, processing now...")
        try:
            dataset.process()
        except Exception as e:
            logger.error(f"Failed to process dataset static files: {e}")
            return

    # Load static graph to device
    data = dataset.get(0).to(device)
    
    # 3. Prepare CSV File
    # If file doesn't exist, write header. If exists, append.
    # But for 'clean' inference of a model, we might want to start fresh or filter.
    # Strategy: Write to a temporary file first, then merge? 
    # Simpler Strategy: Just append to the main file immediately.
    
    file_exists = os.path.exists(output_csv)
    
    # Define columns
    columns = ['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']
    
    # Identify events
    target_dir = Path(dataset.raw_dir)
    event_folders = sorted([d.name for d in target_dir.iterdir() if d.is_dir() and 'event' in d.name])
    logger.info(f"Found {len(event_folders)} events for Model {model_id} in {split_mode} set")
    
    # Check existing events in CSV to avoid duplicates if re-running
    processed_events = set()
    if file_exists:
        # Read only model_id and event_id columns to save memory
        try:
            df_meta = pd.read_csv(output_csv, usecols=['model_id', 'event_id'])
            # Filter for current model
            existing = df_meta[df_meta['model_id'] == model_id]
            processed_events = set(existing['event_id'].unique())
            logger.info(f"Skipping {len(processed_events)} already processed events.")
        except Exception:
            logger.warning("Could not read existing CSV metadata. Assuming fresh start for this model.")

    # 4. Inference Loop
    total_rows_written = 0
    
    # Open CSV in append mode
    # buffer_size=0 ensures immediate write, though buffering is usually fine
    with open(output_csv, 'a') as f:
        # If file is new, write header
        if not file_exists:
            f.write(','.join(columns) + '\n')
            
        with torch.no_grad():
            for event_name in tqdm(event_folders, desc=f"Model {model_id} Inference"):
                try:
                    event_id = int(event_name.split('_')[-1])
                except:
                    continue
                
                # Skip if already done
                if event_id in processed_events:
                    continue

                try:
                    event_data = dataset.load_event(event_name)
                except Exception as e:
                    logger.error(f"⚠️ Error loading {event_name}: {e}")
                    continue

                manhole_seq = event_data['manhole'].to(device)
                cell_seq = event_data['cell'].to(device)
                T = manhole_seq.shape[0]
                
                h_dict = None
                manhole_dyn_t = manhole_seq[0].clone()
                cell_dyn_t = cell_seq[0].clone()
                
                # Buffer for current event results
                event_rows = []
                
                # Inner loop with progress bar
                for t in tqdm(range(T - 1), desc=f"  Ev {event_id}", leave=False):
                    pred_dict, h_dict = model(data, manhole_dyn_t, cell_dyn_t, h_dict)
                    
                    # Predictions to CPU
                    manhole_preds = pred_dict['manhole'].squeeze(-1).cpu().numpy()
                    cell_preds = pred_dict['cell'].squeeze(-1).cpu().numpy()
                    
                    # --- Collect Rows (Optimized) ---
                    # 1D Nodes
                    orig_indices_1d = data['manhole'].orig_idx.cpu().numpy()
                    for idx, val in zip(orig_indices_1d, manhole_preds):
                        # Placeholder row_id (-1), will fix later or ignore
                        event_rows.append(f"-1,{model_id},{event_id},1,{idx},{val:.4f}")
                        
                    # 2D Nodes
                    orig_indices_2d = data['cell'].orig_idx.cpu().numpy()
                    for idx, val in zip(orig_indices_2d, cell_preds):
                        event_rows.append(f"-1,{model_id},{event_id},2,{idx},{val:.4f}")
                    
                    # Autoregressive Update
                    if t < T - 2:
                        manhole_dyn_t = manhole_seq[t+1].clone()
                        cell_dyn_t = cell_seq[t+1].clone()
                        manhole_dyn_t[:, 0] = pred_dict['manhole'].squeeze(-1)
                        cell_dyn_t[:, 1] = pred_dict['cell'].squeeze(-1)
                
                # Write event rows to file immediately
                if event_rows:
                    f.write('\n'.join(event_rows) + '\n')
                    total_rows_written += len(event_rows)
                
                # Cleanup Memory
                del event_rows, manhole_seq, cell_seq, h_dict, pred_dict
                torch.cuda.empty_cache()
                gc.collect()

    logger.info(f"Inference complete. Added {total_rows_written} rows.")

def post_process_submission(output_csv: str):
    """Sort and assign correct row_ids for the final submission."""
    logger.info("Post-processing submission file (Sorting & Re-indexing)...")
    
    # This might still be heavy if file is huge, but much better than holding tensors
    try:
        # Chunk processing could be better, but sorting requires full data.
        # Assuming the final CSV fits in RAM (it's just text/numbers, no graphs).
        df = pd.read_csv(output_csv)
        
        # Sort: Model -> Event -> Node Type -> Node ID -> Timestep (implied by file order? No, need explicit)
        # Wait, previous logic didn't save timestep index. 
        # Kaggle requires ordering by timestep? 
        # "rows are arranged in ascending order of the timesteps by default"
        
        # NOTE: My stream writer didn't save timestep_index to save space/logic complexity.
        # But pandas read might shuffle? No, CSV read preserves order.
        # Since we wrote t=0, t=1... sequentially, the file IS ordered by timestep locally per event.
        # We just need to ensure Model/Event order.
        
        # Actually, to be safe, let's just rely on the write order if we ran Model 1 then Model 2.
        # But strictly, we should assign row_id.
        
        # Let's just reset the row_id column strictly from 0 to N
        df['row_id'] = range(len(df))
        
        # Save back
        df.to_csv(output_csv, index=False)
        logger.info(f"✅ Final submission saved to {output_csv}")
        
    except Exception as e:
        logger.error(f"Post-processing failed (Low RAM?): {e}")
        logger.warning("The submission file exists but row_ids might be -1. You may need a stronger machine to sort it.")

def main():
    parser = argparse.ArgumentParser(description="Inference for UrbanFloodBench (Memory Optimized)")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--model_id", type=int, required=True, 
                        help="Target Model ID (1 or 2)")
    parser.add_argument("--output", type=str, default="submission.csv", 
                        help="Path to the submission CSV file")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run Stream Inference
    run_inference_stream(args.checkpoint, args.model_id, args.output, device)
    
    # Optional: Clean up row_ids at the end
    # Only run this if you are sure this is the LAST model you are running
    # Or run a separate script to fix row_ids later
    # post_process_submission(args.output) 

if __name__ == "__main__":
    main()