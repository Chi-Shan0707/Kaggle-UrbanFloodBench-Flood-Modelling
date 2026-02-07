import torch

# 替换成你实际的 checkpoint 路径
ckpt_path = "checkpoints/best_model.pt" 

try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']
    
    print(f"检查模型文件: {ckpt_path}")
    print("-" * 30)
    
    # 检查关键的归一化参数
    if 'man_dyn_std' in state_dict:
        std_val = state_dict['man_dyn_std']
        print(f"Manhole Std (应该是 16.8 左右): \n{std_val}")
        
        if torch.allclose(std_val, torch.ones_like(std_val)):
            print("\n❌ 破案了！Std 全是 1.0。")
            print("结论：你的归一化参数没存进去，训练代码里肯定把模型重置了！")
        else:
            print("\n✅ Std 数值正常 (不是1.0)。")
            print("如果是这样还爆炸，那可能是 inference.py 加载的问题。")
    else:
        print("❌ 并没有在 checkpoint 里找到 man_dyn_std，可能是旧模型架构！")

except Exception as e:
    print(f"读取失败: {e}")