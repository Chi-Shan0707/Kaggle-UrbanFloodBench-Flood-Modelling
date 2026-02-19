import csv
import os
import sys
import numpy as np
import gc
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 待合并的碎片预测文件
INPUT_FILES = ["submission_1.csv", "submission_2.csv"]

# 2. 中间合并文件路径
MERGED_TEMP_FILE = "merged_intermediate.csv"

# 3. Kaggle 官方提供的样例提交文件 (模板)
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

# 4. 最终生成的提交文件
FINAL_OUTPUT_FILE = "final_submission_filled#005.csv"

# 必须包含的列名
REQUIRED_COLS = ['model_id', 'event_id', 'node_type', 'node_id', 'water_level']
# ===========================================

def merge_inputs(input_files, output_file):
    """阶段 1: 将多个预测 CSV 合并为一个流式中间文件并重置 row_id"""
    print(f"--- 阶段 1/2: 合并碎片文件 ---")
    current_row_id = 0
    headers = ['row_id'] + REQUIRED_COLS
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)
        
        for file_path in input_files:
            if not os.path.exists(file_path):
                print(f"⚠️ 警告：找不到文件 {file_path}，跳过...")
                continue
            
            print(f"正在读取: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in)
                # 检查列是否完整
                if not all(col in reader.fieldnames for col in REQUIRED_COLS):
                    print(f"❌ 错误：{file_path} 缺少必要列")
                    continue
                
                for row in tqdm(reader, desc="Merging", unit="row"):
                    writer.writerow([
                        current_row_id,
                        row['model_id'], row['event_id'],
                        row['node_type'], row['node_id'],
                        row['water_level']
                    ])
                    current_row_id += 1
    print(f"✅ 合并完成，共计 {current_row_id} 行。\n")

class CompactPredictionStore:
    """高效内存存储：使用 Numpy 数组存储预测值，字典存储偏移量"""
    def __init__(self, pred_file):
        self.pred_file = pred_file
        self.offsets = {}
        self.counts = {}
        self.data_array = None
        self.read_cursors = {}
        self._scan_and_load()

    def _scan_and_load(self):
        # 第一次扫描：统计结构
        print("--- 阶段 2/2: 对齐模板 ---")
        print("正在建立索引...")
        total_rows = 0
        with open(self.pred_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Scanning"):
                key = (int(row['model_id']), int(row['event_id']), 
                       int(row['node_type']), int(row['node_id']))
                self.counts[key] = self.counts.get(key, 0) + 1
                total_rows += 1
        
        # 计算偏移量并分配内存
        current_offset = 0
        for key, count in self.counts.items():
            self.offsets[key] = current_offset
            current_offset += count
        
        self.data_array = np.zeros(total_rows, dtype=np.float32)
        
        # 第二次扫描：填充数据
        print(f"加载数据到内存 (占用约 {current_offset * 4 / 1024 / 1024:.1f} MB)...")
        write_cursors = {k: 0 for k in self.counts.keys()}
        with open(self.pred_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Loading"):
                key = (int(row['model_id']), int(row['event_id']), 
                       int(row['node_type']), int(row['node_id']))
                abs_pos = self.offsets[key] + write_cursors[key]
                val = float(row['water_level'])
                self.data_array[abs_pos] = val
                write_cursors[key] += 1
        
        self.read_cursors = {k: 0 for k in self.counts.keys()}
        gc.collect()

    def get_val(self, m, e, t, n):
        key = (m, e, t, n)
        if key not in self.offsets: return None
        cursor = self.read_cursors[key]
        if cursor >= self.counts[key]: return None
        val = self.data_array[self.offsets[key] + cursor]
        self.read_cursors[key] += 1
        return val

def final_fill(store, sample_file, output_file):
    """根据模板生成最终提交文件"""
    print("正在生成最终对齐文件...")
    filled, missing, nans = 0, 0, 0
    
    with open(sample_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in tqdm(reader, desc="Filling"):
            m, e, t, n = int(row['model_id']), int(row['event_id']), \
                         int(row['node_type']), int(row['node_id'])
            
            val = store.get_val(m, e, t, n)
            
            # 处理无效值和缺失值
            if val is not None and (np.isnan(val) or np.isinf(val)):
                val = 0.0
                nans += 1
            
            if val is not None:
                row['water_level'] = f"{val:.4f}"
                filled += 1
            else:
                row['water_level'] = "0.0"
                missing += 1
            
            writer.writerow(row)
            
    print(f"\n任务完成！统计信息:")
    print(f" - 成功填充: {filled}")
    print(f" - 缺失补零: {missing}")
    print(f" - 无效值(NaN)修正: {nans}")
    print(f"最终提交文件: {output_file}")

if __name__ == "__main__":
    # 执行流
    merge_inputs(INPUT_FILES, MERGED_TEMP_FILE)
    
    if os.path.exists(SAMPLE_SUBMISSION_FILE):
        data_store = CompactPredictionStore(MERGED_TEMP_FILE)
        final_fill(data_store, SAMPLE_SUBMISSION_FILE, FINAL_OUTPUT_FILE)
    else:
        print(f"❌ 未找到模板文件 {SAMPLE_SUBMISSION_FILE}，仅完成合并。")