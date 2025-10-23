import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorTransformSafe:
    """安全版 TensorTransform：支持自动计算参数和手动覆盖"""
    def __init__(self, padding=0):
        self.padding = padding

    def _get_pool_params(self, A_shape, B_shape, stride=None, kernel_size=None, padding=None):
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape
        padding = self.padding if padding is None else padding

        if C_in != C_out:
            raise ValueError(f"池化无法改变通道数: C_in={C_in}, C_out={C_out}")

        # 自动计算
        if kernel_size is None:
            kH = H_in // H_out
            kW = W_in // W_out
            if kH <= 0 or kW <= 0:
                raise ValueError(f"池化 kernel_size <=0: {kH}, {kW}")
        else:
            kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        # 自动计算 stride
        if stride is None:
            sH, sW = kH, kW
        else:
            sH, sW = (stride, stride) if isinstance(stride, int) else stride

        return {'kernel_size': (kH, kW), 'stride': (sH, sW), 'padding': padding}

    def _get_conv_params(self, A_shape, B_shape, stride=None, kernel_size=None, padding=None):
        C_in, H_in, W_in = A_shape
        C_out, H_out, W_out = B_shape
        stride = 1 if stride is None else stride
        padding = self.padding if padding is None else padding

        # 自动计算 kernel_size
        if kernel_size is None:
            kH = H_in + 2*padding - (H_out - 1) * stride
            kW = W_in + 2*padding - (W_out - 1) * stride
        else:
            kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        if kH <= 0 or kW <= 0:
            raise ValueError(f"卷积 kernel 尺寸非法: ({kH}, {kW})")

        return {
            'in_channels': C_in,
            'out_channels': C_out,
            'kernel_size': (kH, kW) if kH != kW else kH,
            'stride': (stride, stride) if isinstance(stride, int) else stride,
            'padding': padding
        }

    def apply(self, x, B_shape, op_type='conv', allow_upsample=False, **kwargs):
        """
        x: 输入张量
        B_shape: 目标 shape (C_out, H_out, W_out)
        op_type: 'conv', 'maxpool', 'avgpool'
        kwargs: 可选覆盖参数，例如 kernel_size, stride, padding
        """
        A_shape = x.shape[1:]

        if op_type == 'conv':
            H_in, W_in = A_shape[1], A_shape[2]
            H_out, W_out = B_shape[1], B_shape[2]

            # 上采样模式
            if (H_out > H_in or W_out > W_in) and allow_upsample:
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
                conv_params = self._get_conv_params(A_shape, B_shape,
                                                    stride=kwargs.get('stride'),
                                                    kernel_size=kwargs.get('kernel_size'),
                                                    padding=kwargs.get('padding'))
                conv_layer = nn.Conv2d(**conv_params)
                return conv_layer(x)

        elif op_type == 'maxpool':
            pool_params = self._get_pool_params(A_shape, B_shape,
                                                stride=kwargs.get('stride'),
                                                kernel_size=kwargs.get('kernel_size'),
                                                padding=kwargs.get('padding'))
            return F.max_pool2d(x, **pool_params)

        elif op_type == 'avgpool':
            pool_params = self._get_pool_params(A_shape, B_shape,
                                                stride=kwargs.get('stride'),
                                                kernel_size=kwargs.get('kernel_size'),
                                                padding=kwargs.get('padding'))
            return F.avg_pool2d(x, **pool_params)

        else:
            raise ValueError("op_type must be 'conv', 'maxpool', or 'avgpool'")
