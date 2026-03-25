import numpy as np
import torch
import random

# 固定所有随机种子（关键！）
def set_seed(seed=42):
    random.seed(seed)          # 固定Python原生random
    np.random.seed(seed)       # 固定numpy随机
    torch.manual_seed(seed)    # 固定CPU上的PyTorch随机
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 固定GPU上的PyTorch随机
        torch.cuda.manual_seed_all(seed)  # 多GPU场景