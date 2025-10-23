# ===================== 分类模型 =====================
import torch
from torch import nn

from core import TensorTransformSafe


import torch
import torch.nn as nn
import torch.nn.functional as F

class DeclarClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.ttp = TensorTransformSafe(padding=0)
        self.relu = nn.ReLU()

        # 声明式卷积 & 池化
        self.B_conv1 = (16, 32, 32)
        self.B_pool1 = (16, 16, 16)
        self.B_conv2 = (32, 16, 16)
        self.B_pool2 = (32, 8, 8)
        self.B_conv3 = (64, 8, 8)

        # ✅ 最后一层：原生 1x1 卷积
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.ttp.apply(x, self.B_conv1, op_type='conv')
        x = self.relu(x)
        x = self.ttp.apply(x, self.B_pool1, op_type='maxpool')

        x = self.ttp.apply(x, self.B_conv2, op_type='conv')
        x = self.relu(x)
        x = self.ttp.apply(x, self.B_pool2, op_type='maxpool')

        x = self.ttp.apply(x, self.B_conv3, op_type='conv')
        x = self.relu(x)

        # ✅ 原生 1x1 卷积映射为类别输出
        x = self.final_conv(x)

        # Global Average Pooling -> [B, num_classes]
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        return x



# ===================== 测试 =====================
if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    model = DeclarClassifier(num_classes=10)
    y = model(x)
    print("输出 shape:", y.shape)