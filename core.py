import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorTransformSafe:
    """
    🚀 DeclarCNN - 声明式神经网络构建器
    自动计算卷积 / 池化参数，并允许上采样与拼接。
    """

    def __init__(self, padding=0):
        self.padding = padding

    def _get_pool_params(self, A_shape, B_shape, padding=None):
        padding = self.padding if padding is None else padding
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape

        if C_in != C_out:
            raise ValueError(f"池化无法改变通道数: C_in={C_in}, C_out={C_out}")

        kH = H_in + 2 * padding - (H_out - 1)
        kW = W_in + 2 * padding - (W_out - 1)

        if kH <= 0 or kW <= 0:
            raise ValueError(f"非法池化尺寸: H_in={H_in}, W_in={W_in}, H_out={H_out}, W_out={W_out}")

        return {'kernel_size': (kH, kW), 'stride': (kH, kW), 'padding': padding}

    def _get_conv_params(self, A_shape, B_shape, padding=None):
        padding = self.padding if padding is None else padding
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape
        stride = 1

        kH = H_in + 2 * padding - (H_out - 1) * stride
        kW = W_in + 2 * padding - (W_out - 1) * stride
        if kH <= 0 or kW <= 0:
            raise ValueError(f"卷积 kernel 尺寸非法: ({kH}, {kW})")

        return {
            'in_channels': C_in,
            'out_channels': C_out,
            'kernel_size': (kH, kW) if kH != kW else kH,
            'stride': stride,
            'padding': padding
        }

    def apply(self, x, B_shape, op_type='conv', allow_upsample=False, padding=None):
        A_shape = x.shape[1:]
        pad = self.padding if padding is None else padding

        if op_type == 'conv':
            H_in, W_in = A_shape[1], A_shape[2]
            H_out, W_out = B_shape[1], B_shape[2]

            # 上采样情况
            if (H_out > H_in or W_out > W_in) and allow_upsample:
                x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
                conv_layer = nn.Conv2d(A_shape[0], B_shape[0], kernel_size=1)
                return conv_layer(x)
            else:
                conv_params = self._get_conv_params(A_shape, B_shape, padding=pad)
                conv_layer = nn.Conv2d(**conv_params)
                return conv_layer(x)

        elif op_type in ['maxpool', 'avgpool']:
            pool_params = self._get_pool_params(A_shape, B_shape, padding=pad)
            if op_type == 'maxpool':
                return F.max_pool2d(x, **pool_params)
            else:
                return F.avg_pool2d(x, **pool_params)

        else:
            raise ValueError("op_type 必须是 'conv', 'maxpool', 'avgpool'")

    def concat(self, tensors, dim=1):
        """ 拼接多个张量 """
        return torch.cat(tensors, dim=dim)
