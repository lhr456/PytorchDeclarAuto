import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorTransformSmart:
    """
    智能 kernel 选择版：
    - 允许多解；
    - 优先选择 1x1、3x3；
    - 否则取最小合法整数 kernel；
    - 不满足几何约束则抛异常。
    """

    def __init__(self, padding=0, stride=1):
        self.padding = padding
        self.stride = stride

    def _get_pool_params(self, A_shape, B_shape):
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape

        if C_in != C_out:
            raise ValueError(f"池化无法改变通道数: C_in={C_in}, C_out={C_out}")

        if H_in % H_out != 0 or W_in % W_out != 0:
            raise ValueError(f"池化尺寸不整除: H_in={H_in}, H_out={H_out}")

        kH = H_in // H_out
        kW = W_in // W_out
        return {'kernel_size': (kH, kW), 'stride': (kH, kW), 'padding': self.padding}

    def _get_conv_params(self, A_shape, B_shape):
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape
        pad, stride = self.padding, self.stride

        # 假设 stride 固定，kernel 必须为整数
        kH = H_in + 2 * pad - (H_out - 1) * stride
        kW = W_in + 2 * pad - (W_out - 1) * stride

        # 检查整数合法性
        kH, kW = int(kH), int(kW)
        if kH <= 0 or kW <= 0:
            raise ValueError(f"卷积 kernel 尺寸非法: ({kH}, {kW})")

        # 智能选择：如果存在多解，优先 1x1 或 3x3
        candidates = sorted({kH, kW})
        best_k = None
        if 1 in candidates:
            best_k = 1
        elif 3 in candidates:
            best_k = 3
        else:
            best_k = min(candidates)

        return {
            'in_channels': C_in,
            'out_channels': C_out,
            'kernel_size': best_k,
            'stride': stride,
            'padding': pad
        }

    def apply(self, x, B_shape, op_type='conv', allow_upsample=False):
        A_shape = x.shape[1:]

        if op_type == 'conv':
            H_in, W_in = A_shape[1], A_shape[2]
            H_out, W_out = B_shape[1], B_shape[2]

            if (H_out > H_in or W_out > W_in) and allow_upsample:
                x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
                conv_layer = nn.Conv2d(A_shape[0], B_shape[0], kernel_size=1, stride=1, padding=0)
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
            raise ValueError("op_type 必须是 'conv' / 'maxpool' / 'avgpool'")

    def concat(self, tensors, dim=1):
        return torch.cat(tensors, dim=dim)




if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    ttp = TensorTransformSmart()

    # 常规 3x3
    y1 = ttp.apply(x, (16, 32, 32), op_type='conv')
    print("✅ 3x3 卷积:", y1.shape)

    # 退化到 1x1（自动检测）
    y2 = ttp.apply(y1, (8, 32, 32), op_type='conv')
    print("✅ 1x1 卷积:", y2.shape)

    # 稀有尺寸 → 选最小合法 kernel
    y3 = ttp.apply(x, (8, 28, 28), op_type='conv')
    print("✅ 选最小 kernel:", y3.shape)

