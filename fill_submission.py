import csv
import os
import sys
import numpy as np
import gc
from tqdm import tqdm

# ================= 配置区域 =================
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"
PREDICTION_FILE = "final_submission.csv"
OUTPUT_FILE = "final_submission_filled.csv"
# ===========================================

class CompactPredictionStore:
    def __init__(self, pred_file):
        self.pred_file = pred_file
        self.offsets = {}      # Key -> Start Index in big array
        self.counts = {}       # Key -> Total count of predictions
        self.data_array = None # The giant flat array
        self.read_cursors = {} # Key -> How many we have read so far
        
        # 1. 第一遍扫描：统计每个 Key 的数据量，预计算偏移量
        self._scan_offsets()
        
        # 2. 第二遍扫描：加载数据到 numpy 数组
        self._load_data()

    def _get_key(self, row):
        # Key: (model_id, event_id, node_type, node_id)
        # 使用元组作为字典键
        return (int(row[1]), int(row[2]), int(row[3]), int(row[4]))

    def _scan_offsets(self):
        print("Phase 1/3: 扫描预测文件结构...")
        total_rows = 0
        
        with open(self.pred_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            
            # 自动检测列索引，防止列顺序变化
            try:
                # 寻找必要的列索引
                idx_map = {name: i for i, name in enumerate(header)}
                col_indices = [
                    idx_map['row_id'], 
                    idx_map['model_id'], 
                    idx_map['event_id'], 
                    idx_map['node_type'], 
                    idx_map['node_id'], 
                    idx_map['water_level']
                ]
            except KeyError as e:
                print(f"❌ 错误: 预测文件缺少列 {e}")
                sys.exit(1)

            # 快速遍历统计
            for row in tqdm(reader, desc="Counting", unit="rows"):
                key = (int(row[col_indices[1]]), int(row[col_indices[2]]), 
                       int(row[col_indices[3]]), int(row[col_indices[4]]))
                
                self.counts[key] = self.counts.get(key, 0) + 1
                total_rows += 1
        
        print(f"  - 发现 {total_rows} 个预测点，涉及 {len(self.counts)} 个唯一节点序列。")
        
        # 计算偏移量 (Cumulative Sum)
        current_offset = 0
        for key, count in self.counts.items():
            self.offsets[key] = current_offset
            current_offset += count
            
        self.total_capacity = current_offset
        
        # 预分配 Numpy 数组 (float32 节省一半内存)
        # 5000万个数据点只需要 ~200MB 内存
        print(f"  - 分配内存: {self.total_capacity * 4 / 1024 / 1024:.2f} MB")
        self.data_array = np.zeros(self.total_capacity, dtype=np.float32)

    def _load_data(self):
        print("Phase 2/3: 加载数据到内存...")
        
        # 临时的写入指针
        write_cursors = {k: 0 for k in self.counts.keys()}
        
        with open(self.pred_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            idx_map = {name: i for i, name in enumerate(header)}
            # model, event, type, node, water
            c_m, c_e, c_t, c_n, c_w = (idx_map['model_id'], idx_map['event_id'], 
                                       idx_map['node_type'], idx_map['node_id'], 
                                       idx_map['water_level'])

            for row in tqdm(reader, total=self.total_capacity, desc="Loading", unit="rows"):
                key = (int(row[c_m]), int(row[c_e]), int(row[c_t]), int(row[c_n]))
                val = float(row[c_w])
                
                # 计算在扁平数组中的绝对位置
                # Pos = Start_Offset + Current_Write_Index
                abs_pos = self.offsets[key] + write_cursors[key]
                self.data_array[abs_pos] = val
                
                write_cursors[key] += 1
        
        # 初始化读取指针供后续使用
        self.read_cursors = {k: 0 for k in self.counts.keys()}
        del write_cursors
        gc.collect()

    def get_next_value(self, model, event, n_type, node):
        key = (model, event, n_type, node)
        
        # 检查 Key 是否存在
        if key not in self.offsets:
            return None
        
        # 检查是否还有剩余数据
        cursor = self.read_cursors[key]
        if cursor >= self.counts[key]:
            return None # 数据用完了
            
        # 获取数据
        abs_pos = self.offsets[key] + cursor
        val = self.data_array[abs_pos]
        
        # 指针前移
        self.read_cursors[key] += 1
        return val

def fill_submission():
    if not os.path.exists(PREDICTION_FILE):
        print("❌ 找不到预测文件")
        return

    # 1. 初始化优化的数据存储
    store = CompactPredictionStore(PREDICTION_FILE)
    
    # 2. 填充模板
    print("Phase 3/3: 填充模板...")
    
    missing_count = 0
    nan_count = 0
    filled_count = 0
    
    with open(SAMPLE_SUBMISSION_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        # 估算行数
        total_lines = None
        try:
            total_lines = sum(1 for _ in open(SAMPLE_SUBMISSION_FILE, 'rb')) - 1
        except: pass
        
        for row in tqdm(reader, total=total_lines, desc="Filling"):
            m = int(row['model_id'])
            e = int(row['event_id'])
            t = int(row['node_type'])
            n = int(row['node_id'])
            
            # 从我们的紧凑存储中获取下一个值
            val = store.get_next_value(m, e, t, n)
            
            # 【关键修改 1】检查是否为 NaN (模型算炸了的情况)
            if val is not None and (np.isnan(val) or np.isinf(val)):
                val = 0.0  # 强制修正为 0
                nan_count += 1

            if val is not None:
                row['water_level'] = f"{val:.4f}"
                filled_count += 1
            else:
                # 【关键修改 2】如果没有预测值，显式填 0，防止保留模板里的空值
                row['water_level'] = "0.0"
                missing_count += 1
            
            writer.writerow(row)

    print("\n" + "="*40)
    print(f"✅ 完成！文件已保存: {OUTPUT_FILE}")
    print(f"📊 统计:")
    print(f"   - 成功填充: {filled_count} 行")
    print(f"   - 缺失数据 (已补0): {missing_count} 行")
    print(f"   - 模型 NaN (已补0): {nan_count} 行")
    print("="*40)

if __name__ == "__main__":
    fill_submission()