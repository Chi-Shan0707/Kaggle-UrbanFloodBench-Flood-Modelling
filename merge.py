import csv
import os
import sys
from tqdm import tqdm  # 显示进度条，让你知道没卡死

# ================= 配置区域 =================
# 在这里填入你要合并的文件路径（支持无限多个）
INPUT_FILES = [
    "submission.csv",  # 第一个文件
    "submission_plus.csv"   # 第二个文件
]

OUTPUT_FILE = "final_submission.csv"
# ===========================================

def merge_csv_stream(input_files, output_file):
    print(f"准备合并以下文件到 {output_file}:")
    for f in input_files:
        print(f"  - {f}")
        if not os.path.exists(f):
            print(f"❌ 错误：找不到文件 {f}")
            return

    # Kaggle 提交要求的列顺序
    # 必须确保这一行是你 csv 里实际的列名（除了 row_id 会被重置）
    headers = ['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']
    
    current_row_id = 0
    
    # 使用 buffer_size 优化写入速度
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        
        # 1. 写入表头
        writer.writerow(headers)
        
        # 2. 逐个文件处理
        for file_path in input_files:
            print(f"\n正在流式处理: {file_path} ...")
            
            # 估算行数用于进度条（不加载文件）
            try:
                # 这是一个快速估算行数的方法，如果是 Linux 系统
                # 如果报错或是 Windows，tqdm 会自动降级为不显示总数模式
                total_lines = sum(1 for _ in open(file_path, 'rb')) - 1
            except:
                total_lines = None

            with open(file_path, 'r', encoding='utf-8') as f_in:
                # 使用 DictReader 自动识别列名，防止列顺序不一致
                reader = csv.DictReader(f_in)
                
                # 检查输入文件是否包含必要的列（除了 row_id）
                # 只要有数据列即可，row_id 我们会覆盖
                required_cols = ['model_id', 'event_id', 'node_type', 'node_id', 'water_level']
                if not all(col in reader.fieldnames for col in required_cols):
                    print(f"❌ 错误：文件 {file_path} 缺少必要的列！")
                    print(f"   现有列: {reader.fieldnames}")
                    return

                # 逐行读取，修改 row_id，逐行写入
                # 内存占用极低，只存当前这一行
                for row in tqdm(reader, total=total_lines, unit="row"):
                    writer.writerow([
                        current_row_id,
                        row['model_id'],
                        row['event_id'],
                        row['node_type'],
                        row['node_id'],
                        row['water_level']
                    ])
                    current_row_id += 1

    print("\n" + "="*40)
    print(f"✅ 合并完成！")
    print(f"📄 输出文件: {output_file}")
    print(f"🔢 总行数 (row_id): 0 到 {current_row_id - 1}")
    print("="*40)

if __name__ == "__main__":
    # 检查是否安装 tqdm


    merge_csv_stream(INPUT_FILES, OUTPUT_FILE)