"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-27 02:35:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-27 02:45:29
"""
import numpy as np
import torch
import torch.nn.functional as F
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import nn
from torch.nn import Parameter, init
from torch.types import Device

__all__ = ["MLGConv2d"]


class MLGConv2d(nn.Module):
    """
    description: Conv2d layer with memory-efficient multi-level generation (MLG)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        w_bit: int = 16,
        in_bit: int = 16,
        device: Device = torch.device("cuda"),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.w_bit = w_bit
        self.qb = w_bit
        self.qu = w_bit
        self.qv = w_bit
        self.in_bit = in_bit
        self.device = device

        ### allocate parameters
        self.weight = None
        ### build trainable parameters
        self.build_parameters()

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="normal", device=self.device)

        self.weight_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_sym")
        self.basis_quantizer = weight_quantize_fn(self.qb, alg="dorefa_sym")
        self.coeff_in_quantizer = weight_quantize_fn(self.qu, alg="dorefa_sym")
        self.coeff_out_quantizer = weight_quantize_fn(self.qv, alg="dorefa_sym")

        ### default set to slow forward
        self.disable_fast_forward()
        ### default disable dynamic weight generation
        self.disable_dynamic_weight()
        self.eye_b = None
        self.eye_v = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

    def build_parameters(self):
        self.weight = Parameter(
            torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            .to(self.device)
            .float()
        )

    def reset_parameters(self):
        init.kaiming_normal_(self.weight.data, mode="fan_out", nonlinearity="relu")

        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size ** 2
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def build_weight(self):
        if self.w_bit < 16:
            if self.dynamic_weight_flag:
                if self.coeff_in is not None:
                    coeff_in = self.coeff_in_quantizer(self.coeff_in)
                    self.basis = self.weight[
                        : self.base_out, : self.base_in, ...
                    ]  # [base_out, base_in, k, k]
                else:
                    coeff_in = None
                    self.basis = self.weight[: self.base_out, ...]  # [base_out, inc, k, k]
                basis = self.basis_quantizer(self.basis)
                if self.coeff_out is not None:
                    coeff_out = self.coeff_out_quantizer(self.coeff_out)
                else:
                    coeff_out = None
                weight = self.weight_generation(basis, coeff_in, coeff_out)
            else:
                weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight
            if self.dynamic_weight_flag:
                if self.coeff_in is not None:
                    self.basis = weight[: self.base_out, : self.base_in, ...]  # [base_out, base_in, k, k]
                else:
                    self.basis = weight[: self.base_out, ...]  # [base_out, inc, k, k]

                weight = self.weight_generation(self.basis, self.coeff_in, self.coeff_out)

        return weight

    def enable_fast_forward(self):
        self.fast_forward_flag = True

    def disable_fast_forward(self):
        self.fast_forward_flag = False

    def load_parameters(self, param_dict):
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)

    def enable_dynamic_weight(self, base_in, base_out):
        ### multi-level weight generation
        self.base_in = base_in  # input channel base
        self.base_out = base_out  # output channel base

        if base_out == 0:  ## disable cross-kernel generation
            self.base_out = self.out_channels  ## maximum
        elif min(self.out_channels, self.in_channels * self.kernel_size ** 2) > self.base_out > 0:
            ### enable generation
            self.base_out = base_out
        else:
            ### base_out is too large, cannot save param, then disable it
            self.base_out = self.out_channels

        ### only when base_in < min(in_channel, kernel_size**2), will intra-kernel generation save #params.
        if min(self.in_channels, self.kernel_size ** 2) > self.base_in > 0:
            self.coeff_in = Parameter(
                torch.Tensor(self.base_out, self.in_channels, self.base_in).to(self.device)
            )
            # init.xavier_normal_(self.coeff_in)
            init.kaiming_normal_(self.coeff_in, mode="fan_out", nonlinearity="relu")
        else:
            ### base_in >= min(in_channel, kernel_size**2), will use the original weight
            self.coeff_in = None
        ### onlt when base_out < min(out_channel, in_channel*kernel_size**2), will cross-kernel generation save #params.
        if min(self.out_channels, self.in_channels * self.kernel_size ** 2) > self.base_out > 0:
            self.coeff_out = Parameter(torch.Tensor(self.out_channels, self.base_out).to(self.device))
            init.kaiming_normal_(self.coeff_out, mode="fan_out", nonlinearity="relu")
        else:
            self.coeff_out = None

        self.dynamic_weight_flag = True if self.coeff_in is not None or self.coeff_out is not None else False
        if self.dynamic_weight_flag:
            if self.coeff_in is not None:
                self.basis = self.weight[: self.base_out, : self.base_in, ...]
            else:
                self.basis = self.weight[: self.base_out, ...]
        else:
            self.basis = None

    def disable_dynamic_weight(self):
        self.dynamic_weight_flag = False

    def weight_generation(self, basis, coeff_in, coeff_out):
        ### Level 1
        if coeff_in is not None:
            # weight_1 [base_out, inc, k^2]
            # coeff_in x basis = [bo, inc, bi] x [bo, bi, k^2]
            basis = basis.view(basis.size(0), basis.size(1), -1)
            weight_1 = torch.matmul(coeff_in, basis)
        else:
            weight_1 = basis

        ### Level 2
        if coeff_out is not None:
            # weight_2 [outc, inc*k*k]
            weight_1 = weight_1.view(weight_1.size(0), -1)
            weight_2 = torch.matmul(coeff_out, weight_1)
            weight_2 = weight_2.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            ## do not use self.out_channel, since for dwconv, we should set out_channel to 1 here.
            weight_2 = weight_1.view(
                self.weight.size(0), self.in_channels, self.kernel_size, self.kernel_size
            )
        return weight_2

    def get_output_dim(self, img_height, img_width):
        h_out = (img_height - self.kernel_size + 2 * self.padding) / self.stride + 1
        w_out = (img_width - self.kernel_size + 2 * self.padding) / self.stride + 1
        return (int(h_out), int(w_out))

    def get_num_params(self, fullrank=False):
        if (self.dynamic_weight_flag == True) and (fullrank == False):
            total = self.basis.numel()
            if self.coeff_in is not None:
                total += self.coeff_in.numel()
            if self.coeff_out is not None:
                total += self.coeff_out.numel()
        else:
            total = self.weight.numel()
        if self.bias is not None:
            total += self.bias.numel()

        return total

    def get_param_size(self, fullrank=False):
        total = 0
        if (self.dynamic_weight_flag == True) and (fullrank == False):
            total += self.basis.numel() * self.qb / 8
            if self.coeff_in is not None:
                total += self.coeff_in.numel() * self.qu / 8
            if self.coeff_out is not None:
                total += self.coeff_out.numel() * self.qv / 8
        else:
            total += self.weight.numel() * 4
        if self.bias is not None:
            total += self.bias.numel() * 4
        return total

    def get_ortho_loss(self):
        ### we want row vectors in the basis to be orthonormal
        if self.dynamic_weight_flag:  ### at least one-level generation
            ## basis ortho loss always exists !
            if self.coeff_in is not None and self.coeff_in.size(2) > 1:
                if self.basis.size(1) > 1:
                    ### only penalize when there are at least two row/column vectors
                    basis = self.basis.view(self.basis.size(0), self.basis.size(1), -1)  # [bo, bi, k^2]
                    dot_b = torch.matmul(
                        basis, basis.permute([0, 2, 1])
                    )  # [bo, bi, k^2] x [bo, k^2, bi] = [bo, bi, bi]
                else:
                    dot_b = None
                ## U
                coeff_in = self.coeff_in / (
                    self.coeff_in.data.norm(p=2, dim=1, keepdim=True) + 1e-8
                )  # normalization
                dot_u = torch.matmul(
                    coeff_in.permute(0, 2, 1), coeff_in
                )  # [bo, bi, ci-bi] x [bo, ci-bi, bi] = [bo, bi, bi]
            else:
                dot_u = None

            if self.coeff_out is not None:
                if self.coeff_in is None:
                    ### if there is no intra-kernel generation, only cross-kernel generation, e.g., conv1x1, we have to treat basis as a matrix [bo, cin*k*k] and encourage it to have bo orthogonal rows
                    basis = self.basis.view(self.basis.size(0), -1)  # [bo, ci*k^2]
                    dot_b = torch.matmul(
                        basis, basis.permute([1, 0])
                    )  # [bo, ci*k^2] x [ci*k^2, bo] = [bo, bo]
                # V
                coeff_out = self.coeff_out / (
                    self.coeff_out.data.norm(p=2, dim=0, keepdim=True) + 1e-8
                )  # normalization
                dot_v = torch.matmul(coeff_out.t(), coeff_out)  # [bo, co-bo] x [co-bo, bo] = [bo, bo]
            else:
                dot_v = None
            if self.basis is not None and self.eye_b is None:
                self.eye_b = torch.eye(dot_b.size(-1), dtype=dot_b.dtype, device=dot_b.device)
                if dot_b.ndim > 2:
                    self.eye_b = self.eye_b.unsqueeze(0).repeat(basis.size(0), 1, 1)
            if self.coeff_out is not None and self.eye_v is None:
                self.eye_v = torch.eye(dot_v.size(-1), dtype=dot_v.dtype, device=dot_v.device)
            loss = 0
            if dot_b is not None:
                loss = loss + F.mse_loss(dot_b, self.eye_b)
            if dot_u is not None:
                loss = loss + F.mse_loss(dot_u, self.eye_b)
            if dot_v is not None:
                loss = loss + F.mse_loss(dot_v, self.eye_v)
        else:
            loss = 0

        return loss

    def assign_separate_weight_bit(self, qb, qu, qv, quant_ratio_b=1, quant_ratio_u=1, quant_ratio_v=1):
        qb, qu, qv = min(qb, 32), min(qu, 32), min(qv, 32)
        self.qb, self.qu, self.qv = qb, qu, qv
        self.basis_quantizer = weight_quantize_fn(qb, alg="dorefa_sym")
        self.coeff_in_quantizer = weight_quantize_fn(qu, alg="dorefa_sym")
        self.coeff_out_quantizer = weight_quantize_fn(qv, alg="dorefa_sym")
        self.basis_quantizer.set_quant_ratio(quant_ratio_b)
        self.coeff_in_quantizer.set_quant_ratio(quant_ratio_u)
        self.coeff_out_quantizer.set_quant_ratio(quant_ratio_v)

    def set_quant_ratio(self, quant_ratio_b=1, quant_ratio_u=1, quant_ratio_v=1, quant_ratio_in=1):
        if hasattr(self, "basis_quantizer"):
            self.basis_quantizer.set_quant_ratio(quant_ratio_b)
        if hasattr(self, "coeff_in_quantizer"):
            self.coeff_in_quantizer.set_quant_ratio(quant_ratio_u)
        if hasattr(self, "coeff_out_quantizer"):
            self.coeff_out_quantizer.set_quant_ratio(quant_ratio_v)
        self.input_quantizer.set_quant_ratio(quant_ratio_in)

    def forward(self, x):
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()
        else:
            weight = self.weight
        #### record weight_2
        self.weight_2 = weight

        out = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.padding)

        return out

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, padding={padding}, wb={w_bit}, ib={in_bit}"
        )
        if self.bias is None:
            s += ", bias=False"
        if self.dynamic_weight_flag:
            s += "base_in={base_in}, base_out={base_out}"
        return s.format(**self.__dict__)
