"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-27 02:58:08
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-27 02:59:39
"""

import numpy as np
import torch
import torch.nn.functional as F
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import nn
from torch.nn import Parameter, init
from torch.types import Device

__all__ = ["MLGLinear"]


class MLGLinear(nn.Module):
    """
    description: Linear layer with multi-level generation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        w_bit: int = 16,
        in_bit: int = 16,
        device: Device = torch.device("cuda"),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.device = device

        ### allocate parameters
        self.weight = None
        ### build trainable parameters
        self.build_parameters()

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="normal", device=self.device)

        self.weight_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_sym")

        ### default set to slow forward
        self.disable_fast_forward()
        ### default disable dynamic weight generation
        self.disable_dynamic_weight()
        self.eye_b = None
        self.eye_v = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

    def build_parameters(self):
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features).to(self.device))

    def reset_parameters(self):
        init.kaiming_normal_(self.weight.data)

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def build_weight(self):
        if self.w_bit < 16:
            weight = self.weight_quantizer(self.weight)
            if self.dynamic_weight_flag:
                self.basis = weight[: self.base_out, : self.base_in, ...]  # [base_out, base_in, k]
                if self.coeff_in is not None:
                    coeff_in = self.weight_quantizer(self.coeff_in)
                else:
                    coeff_in = None
                if self.coeff_out is not None:
                    coeff_out = self.weight_quantizer(self.coeff_out)
                else:
                    coeff_out = None
                weight = self.weight_generation(self.basis, coeff_in, coeff_out)
            else:
                weight = weight.view(self.out_features, -1)[:, : self.in_features]
        else:
            weight = self.weight
            if self.dynamic_weight_flag:
                self.basis = weight[: self.base_out, : self.base_in, ...]  # [base_out, base_in, k]
                weight = self.weight_generation(self.basis, self.coeff_in, self.coeff_out)
            else:
                weight = self.weight.view(self.out_features, -1)[:, : self.in_features]

        return weight

    def enable_fast_forward(self):
        self.fast_forward_flag = True

    def disable_fast_forward(self):
        self.fast_forward_flag = False

    def enable_dynamic_weight(self, base_in, base_out):
        ### multi-level weight generation
        self.base_in = base_in  # input channel base
        self.base_out = base_out  # output channel base
        if base_out == 0:  ## disable cross-kernel generation
            self.base_out = self.out_features
        elif min(self.out_features, self.in_features) > self.base_out > 0:
            ### enable generation
            self.base_out = base_out
        else:
            ### base_out is too large, cannot save param, then disable it
            self.base_out = self.out_features

        ### TODO temporarily never turn on intra-kernel generation.
        if int((self.in_features + self.miniblock - 1) / self.miniblock) > self.base_in > 0:
            self.coeff_in = Parameter(
                torch.Tensor(
                    self.base_out,
                    int((self.in_features + self.miniblock - 1) / self.miniblock),
                    self.base_in,
                ).to(self.device)
            )

            init.kaiming_normal_(self.coeff_in, mode="fan_out", nonlinearity="relu")
        else:
            ### input channel is already very small
            self.coeff_in = None
        ### only when base_out < min(out_features, in_features), will cross_out_features generation save #params.
        if min(self.out_features, self.in_features) > self.base_out > 0:
            self.coeff_out = Parameter(torch.Tensor(self.out_features, self.base_out).to(self.device))
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
            # weight_1 [base_out, inc, blocksize]
            # coeff_in x basis = [bo, inc-bi, bi] x [bo, bi, blocksize]
            weight_1 = torch.matmul(coeff_in, basis)

        else:
            weight_1 = basis

        ### Level 2
        if coeff_out is not None:
            ### TODO not supported yet
            # weight_2 [outc, inc]
            weight_1 = weight_1.view(weight_1.size(0), -1)[:, : self.in_features]
            weight_2 = torch.matmul(coeff_out, weight_1)
        else:
            weight_2 = weight_1.view(weight_1.size(0), -1)[:, : self.in_features]

        return weight_2

    def get_num_params(self, fullrank=False):
        if (self.dynamic_weight_flag == True) and (fullrank == False):
            total = self.basis.numel()
            if self.coeff_in is not None:
                total += self.coeff_in.numel()
            if self.coeff_out is not None:
                total += self.coeff_out.numel()
        else:
            total = self.out_features * self.in_features
        if self.bias is not None:
            total += self.bias.numel()

        return total

    def get_param_size(self, fullrank=False, fullprec=False):
        if (self.dynamic_weight_flag == True) and (fullrank == False):
            total = self.basis.numel() * self.w_bit / 8
            if self.coeff_in is not None:
                total += self.coeff_in.numel() * self.w_bit / 8
            if self.coeff_out is not None:
                total += self.coeff_out.numel() * self.w_bit / 8
        else:
            if fullprec:
                total = (self.out_features * self.in_features) * 4
            else:
                total = (self.out_features * self.in_features) * self.w_bit / 8
        if self.bias is not None:
            total += self.bias.numel() * 4
        return total

    def get_ortho_loss(self):
        ### we want row vectors in the basis to be orthonormal
        if self.dynamic_weight_flag:
            if self.coeff_in is not None:
                basis = self.basis.view(self.basis.size(0), self.basis.size(1), -1)  # [bo, bi, k^2]
                ## basis
                dot_b = torch.matmul(
                    basis, basis.permute([0, 2, 1])
                )  # [bo, bi, k^2] x [bo, k^2, bi] = [bo, bi, bi]

                ## U
                coeff_in = self.coeff_in / (
                    self.coeff_in.data.norm(p=2, dim=1, keepdim=True) + 1e-8
                )  # normalization
                dot_u = torch.matmul(
                    coeff_in.permute(0, 2, 1), coeff_in
                )  # [bo, bi, ci-bi] x [bo, ci-bi, bi] = [bo, bi, bi]
            else:
                pass

            if self.coeff_out is not None:
                # V
                coeff_out = self.coeff_out / (
                    self.coeff_out.data.norm(p=2, dim=0, keepdim=True) + 1e-8
                )  # normalization
                dot_v = torch.matmul(coeff_out.t(), coeff_out)  # [bo, co-bo] x [co-bo, bo] = [bo, bo]
            else:
                pass

            if self.coeff_in is not None and self.eye_b is None:
                self.eye_b = (
                    torch.eye(dot_b.size(-1), dtype=dot_b.dtype, device=dot_b.device)
                    .unsqueeze(0)
                    .repeat(basis.size(0), 1, 1)
                )
            if self.coeff_out is not None and self.eye_v is None:
                self.eye_v = torch.eye(dot_v.size(-1), dtype=dot_v.dtype, device=dot_v.device)
            loss = 0
            if self.coeff_in is not None:
                loss = loss + F.mse_loss(dot_b, self.eye_b) + F.mse_loss(dot_u, self.eye_b)
            if self.coeff_out is not None:
                loss = loss + F.mse_loss(dot_v, self.eye_v)
        else:
            loss = 0

        return loss

    def forward(self, x):
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()
        else:
            weight = self.weight  # .view(self.out_features, -1)[:, :self.in_features]

        ### record weight_2
        self.weight_2 = weight
        out = F.linear(x, weight, bias=self.bias)

        return out

    def extra_repr(self):
        s = "{in_features}, {out_features}, wb={w_bit}, ib={in_bit}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)
