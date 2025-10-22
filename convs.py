import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorTransformSafe:
    """
    安全版 TensorTransform：
    - 自动判断卷积/池化是否可行
    - 卷积/池化参数计算保证 kernel_size > 0
    - 如果目标 H/W 大于输入，可以选择使用上采样
    """
    def __init__(self, padding=0):
        self.padding = padding

    def _get_pool_params(self, A_shape, B_shape):
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape

        if C_in != C_out:
            raise ValueError(f"池化无法改变通道数: C_in={C_in}, C_out={C_out}")

        kH = H_in + 2*self.padding - (H_out - 1)
        kW = W_in + 2*self.padding - (W_out - 1)

        if kH <= 0 or kW <= 0:
            raise ValueError(f"池化尺寸非法: H_in={H_in}, W_in={W_in}, H_out={H_out}, W_out={W_out}")

        return {'kernel_size': (kH, kW), 'stride': (kH, kW), 'padding': self.padding}

    def _get_conv_params(self, A_shape, B_shape):
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape
        stride = 1

        kH = H_in + 2*self.padding - (H_out - 1) * stride
        kW = W_in + 2*self.padding - (W_out - 1) * stride

        if kH <= 0 or kW <= 0:
            raise ValueError(
                f"卷积尺寸非法: H_in={H_in}, W_in={W_in}, H_out={H_out}, W_out={W_out}, "
                f"计算得 kernel_size=({kH},{kW})"
            )

        return {
            'in_channels': C_in,
            'out_channels': C_out,
            'kernel_size': (kH, kW) if kH != kW else kH,
            'stride': stride,
            'padding': self.padding
        }

    def apply(self, x, B_shape, op_type='conv', allow_upsample=False):
        A_shape = x.shape[1:]  # 忽略 batch

        if op_type == 'conv':
            H_in, W_in = A_shape[1], A_shape[2]
            H_out, W_out = B_shape[1], B_shape[2]

            # 如果输出比输入大且允许升采样
            if (H_out > H_in or W_out > W_in) and allow_upsample:
                x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
                # 卷积仍然改变通道数
                conv_params = {
                    'in_channels': A_shape[0],
                    'out_channels': B_shape[0],
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0
                }
                conv_layer = nn.Conv2d(**conv_params)
                return conv_layer(x)
            else:
                conv_params = self._get_conv_params(A_shape, B_shape)
                conv_layer = nn.Conv2d(**conv_params)
                return conv_layer(x)

        elif op_type == 'maxpool':
            pool_params = self._get_pool_params(A_shape, B_shape)
            return F.max_pool2d(x, **pool_params)

        elif op_type == 'avgpool':
            pool_params = self._get_pool_params(A_shape, B_shape)
            return F.avg_pool2d(x, **pool_params)

        else:
            raise ValueError("op_type must be 'conv', 'maxpool', or 'avgpool'")

# ========== nn.Module 示例 ==========
class SafeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ttp = TensorTransformSafe(padding=1)
        self.relu = nn.ReLU()

        # 定义目标输出 shape
        self.B_conv1 = (16, 32, 32)


        self.B_conv2 = (16, 14, 14)

        self.B_conv3 = (18, 13,7)
        self.pool1=(18,4,4)

    def forward(self, x):
        # 卷积
        x = self.ttp.apply(x, self.B_conv1, op_type='conv')

        # 卷积
        x = self.ttp.apply(x, self.B_conv2, op_type='conv')
        x = self.ttp.apply(x, self.B_conv3, op_type='conv')

        x = self.ttp.apply(x, self.pool1, op_type='maxpool')
        # 上采样

        # 激活
        x = self.relu(x)
        return x

# 测试
x = torch.randn(1, 3, 32, 32)
model = SafeModel()
out = model(x)
print("最终输出 shape:", out.shape)
