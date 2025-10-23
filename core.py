import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorTransformSafe:
    """安全版 TensorTransform，用于根据目标 shape 自动确定卷积和池化参数"""
    def __init__(self, padding=0):
        self.padding = padding

    def _get_pool_params(self, A_shape, B_shape):
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape

        if C_in != C_out:
            raise ValueError(f"池化无法改变通道数: C_in={C_in}, C_out={C_out}")

        kH = H_in // H_out
        kW = W_in // W_out
        if kH <= 0 or kW <= 0:
            raise ValueError(f"池化 kernel_size <=0: {kH}, {kW}")

        return {'kernel_size': (kH, kW), 'stride': (kH, kW), 'padding': self.padding}

    def _get_conv_params(self, A_shape, B_shape):
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape
        stride = 1
        kH = H_in + 2*self.padding - (H_out - 1) * stride
        kW = W_in + 2*self.padding - (W_out - 1) * stride
        if kH <= 0 or kW <= 0:
            raise ValueError(f"卷积 kernel 尺寸非法: ({kH}, {kW})")

        return {
            'in_channels': C_in,
            'out_channels': C_out,
            'kernel_size': (kH, kW) if kH != kW else kH,
            'stride': stride,
            'padding': self.padding
        }

    def apply(self, x, B_shape, op_type='conv', allow_upsample=False):
        A_shape = x.shape[1:]

        if op_type == 'conv':
            H_in, W_in = A_shape[1], A_shape[2]
            H_out, W_out = B_shape[1], B_shape[2]

            if (H_out > H_in or W_out > W_in) and allow_upsample:
                # 上采样模式
                x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
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