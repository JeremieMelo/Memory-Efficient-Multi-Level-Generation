"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-27 02:34:42
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-27 02:34:42
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.compute import lowrank_decompose
from pyutils.optimizer import RAdam

from .layers import MLGConv2d, MLGLinear

__all__ = ["MLGBaseModel"]


class MLGBaseModel(nn.Module):
    _conv = (MLGConv2d,)
    _linear = (MLGLinear,)
    _conv_linear = (MLGConv2d, MLGLinear)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                m.reset_parameters()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_num_params(self, fullrank=False):
        params = {}
        conv_counter = 1
        bn_counter = 1
        fc_counter = 1
        for layer in self.modules():
            if isinstance(layer, self._conv):
                params[f"conv{conv_counter}"] = layer.get_num_params(fullrank=fullrank)
                conv_counter += 1
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                params[f"bn{bn_counter}"] = layer.weight.numel() + layer.bias.numel()
                bn_counter += 1
            elif isinstance(layer, self._linear):
                params[f"fc{fc_counter}"] = layer.get_num_params(fullrank=fullrank)
                fc_counter += 1

        return params

    def get_total_num_params(self, fullrank=False):
        params = self.get_num_params(fullrank=fullrank)
        return sum(i for i in params.values())

    def get_param_size(self, fullrank=False, fullprec=False):
        params = {}
        conv_counter = 1
        bn_counter = 1
        fc_counter = 1
        for layer in self.modules():
            if isinstance(layer, self._conv):
                params[f"conv{conv_counter}"] = layer.get_param_size(fullrank)
                # w_bit = 32 if fullrank else layer.w_bit
                conv_counter += 1
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                params[f"bn{bn_counter}"] = layer.weight.numel() * 4 + layer.bias.numel() * 4  ## Byte
                bn_counter += 1
            elif isinstance(layer, self._linear):
                # w_bit = 32 if fullrank else layer.w_bit
                params[f"fc{fc_counter}"] = layer.get_param_size(fullrank, fullprec)
                fc_counter += 1

        return params

    def get_total_param_size(self, fullrank=False, fullprec=False):
        if fullrank and fullprec:
            return self.get_total_num_params(fullrank=True) * 4 / 1024
        elif fullrank and not fullprec:
            raise NotImplementedError
        elif not fullrank and fullprec:
            return self.get_total_num_params(fullrank=False) * 4 / 1024
        else:
            params = self.get_param_size(fullrank=False, fullprec=False)
            return sum(i for i in params.values()) / 1024  ## KB

    def get_weight_compression_ratio(self):
        return self.get_total_num_params(fullrank=False) / self.get_total_num_params(fullrank=True)

    def get_memory_compression_ratio(self):
        return self.get_total_param_size(fullrank=False, fullprec=False) / self.get_total_param_size(
            fullrank=True, fullprec=True
        )

    def enable_dynamic_weight(self, base_in, base_out, last_layer=False):
        base_in, base_out = int(base_in), int(base_out)
        ### global basis mode
        for layer in self.modules():
            if isinstance(layer, self._conv):
                if layer.kernel_size == 1 and base_out >= min(layer.out_channels, layer.in_channels):
                    ### when base_out is too large, we want to use relatively large base_out to turn on cross-kernel generation
                    b_out = int(min(layer.out_channels, layer.in_channels) * 0.9)
                    layer.enable_dynamic_weight(base_in, b_out)
                else:
                    layer.enable_dynamic_weight(base_in, base_out)
            elif isinstance(layer, self._linear):
                if last_layer:
                    layer.enable_dynamic_weight(
                        layer.in_features,
                        base_out,
                    )

    def disable_dynamic_weight(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.disable_dynamic_weight()

    def get_ortho_loss(self):
        loss = 0
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                loss = loss + layer.get_ortho_loss()
        return loss

    def get_approximation_loss(self, model, cache=False):
        loss = 0
        ref_modules = list(model.modules())
        for idx, layer in enumerate(self.modules()):
            if isinstance(layer, self._conv_linear):
                if cache:
                    w = layer.weight_2
                else:
                    w = layer.build_weight()

                ref_layer = ref_modules[idx]
                if hasattr(ref_layer, "weight_2") and ref_layer.weight_2 is not None:
                    w_ref = ref_layer.weight_2.data
                else:
                    w_ref = ref_layer.build_weight().data
                    ref_layer.weight_2 = w_ref
                loss = loss + F.mse_loss(w.view(-1), w_ref.view(-1))
        return loss

    def approximate_target_model(self, model, n_step=3000, alg="svd"):
        ### first copy bias and BN
        for m1, m2 in zip(self.modules(), model.modules()):
            if isinstance(m1, nn.BatchNorm2d) or isinstance(m1, nn.GroupNorm):
                m1.weight.data.copy_(m2.weight.data)
                m1.bias.data.copy_(m2.bias.data)
                m1.running_mean.data.copy_(m2.running_mean.data[: m1.num_features])
                m1.running_var.data.copy_(m2.running_var.data[: m1.num_features])
            elif isinstance(m1, self._conv_linear):
                if m1.bias is not None:
                    m1.bias.data.copy_(m2.bias.data)
            elif isinstance(m1, torch.quantization.observer.MovingAverageMinMaxObserver):
                m1.min_val.data.copy_(m2.min_val)
                m1.max_val.data.copy_(m2.max_val)

        if alg == "train":
            optimizer = RAdam((p for p in self.parameters() if p.requires_grad), lr=2e-2)
            for i in range(n_step):
                loss = self.get_approximation_loss(model, cache=False)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 500 == 0:
                    print(f"step {i}: loss: {loss.item():.4f}")
        elif alg == "svd":
            for layer, ref_layer in zip(self.modules(), model.modules()):
                if isinstance(layer, self._conv):
                    layer.weight.data.copy_(ref_layer.weight)
                    if layer.coeff_out is not None:
                        u, v = lowrank_decompose(
                            ref_layer.weight.data.view(ref_layer.out_channels, -1),
                            r=layer.base_out,
                            u_ortho=True,
                        )
                        layer.coeff_out.data.copy_(u)
                    else:
                        v = ref_layer.weight.data
                    if layer.coeff_in is not None:
                        u, v = lowrank_decompose(
                            v.view(layer.base_out, layer.in_channels, -1), r=layer.base_in, u_ortho=False
                        )
                        layer.coeff_in.data.copy_(u)
                        layer.basis.data.copy_(v.view_as(layer.basis.data))
                        if layer.basis is not None:
                            layer.basis.data.copy_(v.view_as(layer.basis.data))
                        else:
                            layer.weight.data.copy_(v)
                elif isinstance(layer, self._linear):
                    layer.weight.data.copy_(ref_layer.weight)
            with torch.no_grad():
                loss = self.get_approximation_loss(model, cache=False)
                print(f"[svd] step 0: loss: {loss.item():.4f}")

    def assign_separate_weight_bit(self, qb, qu, qv, quant_ratio_b=1, quant_ratio_u=1, quant_ratio_v=1):
        for layer in self.modules():
            if isinstance(layer, self._conv):
                layer.assign_separate_weight_bit(
                    qb,
                    qu,
                    qv,
                    quant_ratio_b=quant_ratio_b,
                    quant_ratio_u=quant_ratio_u,
                    quant_ratio_v=quant_ratio_v,
                )

    def set_quant_ratio(self, quant_ratio_b=1, quant_ratio_u=1, quant_ratio_v=1, quant_ratio_in=1):
        for layer in self.modules():
            if isinstance(layer, self._conv):
                layer.set_quant_ratio(quant_ratio_b, quant_ratio_u, quant_ratio_v, quant_ratio_in)

    def init_from_pretrained_model(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
