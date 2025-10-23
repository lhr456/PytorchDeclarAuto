# ===================== 分类模型 =====================
import torch
from torch import nn

from core import TensorTransformSafe


class SafeClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.ttp = TensorTransformSafe(padding=0)
        self.relu = nn.ReLU()

        # 各层目标形状
        self.B_conv1 = (16, 32, 32)   # 输入 3x32x32 → 16x32x32
        self.B_pool1 = (16, 16, 16)   # 池化减半
        self.B_conv2 = (32, 16, 16)   # 通道增加
        self.B_pool2 = (32, 8, 8)     # 再次减半
        self.B_conv3 = (64, 8, 8)     # 通道增加

        # 分类头
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.relu(self.ttp.apply(x, self.B_conv1, op_type='conv'))
        x = self.ttp.apply(x, self.B_pool1, op_type='maxpool')

        x = self.relu(self.ttp.apply(x, self.B_conv2, op_type='conv'))
        x = self.ttp.apply(x, self.B_pool2, op_type='maxpool')

        x = self.relu(self.ttp.apply(x, self.B_conv3, op_type='conv'))

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ===================== 测试 =====================
if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    model = SafeClassifier(num_classes=10)
    y = model(x)
    print("输出 shape:", y.shape)