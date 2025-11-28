from functools import lru_cache
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from combnet.functional import lc_batch_comb
from combnet.filters import *
import torchaudio
from torch.nn.utils.parametrize import register_parametrization
from typing import Union, Optional
from philtorch.lti import lfilter, state_space
from philtorch.mat import companion
from torch_fftconv import fft_conv1d
import math

from combnet.utils import (
    neg_alpha_even_angle2poles,
    pos_alpha_even_angle2poles,
    poles2res,
)

torch.backends.cuda.cufft_plan_cache[0].max_size = 32


# Wrap between K and N
def wrap(x, K, N):
    return ((x - K) % (N - K)) + K


# Range to use for regularization
r = torch.linspace(50, 8000, 1024)
dr = torch.logspace(
    0, -2, 1024
)  # Decay for high frequencies to allow harmonics crossing


class SmoothingCoef(nn.Module):
    def forward(self, x):
        return x.sigmoid()

    def right_inverse(self, y):
        return (y / (1 - y)).log()


class MinMax(SmoothingCoef):
    def __init__(self, min=0.0, max: Union[float, torch.Tensor] = 1.0):
        super().__init__()
        if isinstance(min, torch.Tensor):
            self.register_buffer("min", min, persistent=False)
        else:
            self.min = min

        if isinstance(max, torch.Tensor):
            self.register_buffer("max", max, persistent=False)
        else:
            self.max = max

    def forward(self, x):
        return super().forward(x) * (self.max - self.min) + self.min

    def right_inverse(self, y):
        return super().right_inverse((y - self.min) / (self.max - self.min))


class ScalingFunction(SmoothingCoef):
    def __init__(self, min_freq, max_freq):
        super().__init__()
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.fratio = self.max_freq / self.min_freq

    @torch.compile
    def forward(self, f: torch.Tensor):
        s = super().forward(f)
        o = self.min_freq * self.fratio**s
        return o


class ScalingFunctionBins(ScalingFunction):
    def __init__(self, min_freq, max_freq, min_bin, max_bin):
        super().__init__(min_freq, max_freq)
        self.min_bin = min_bin
        self.max_bin = max_bin

    @torch.compile
    def forward(self, f: torch.Tensor):
        s = F.sigmoid(f)
        nbins = self.max_bin - self.min_bin
        p = nbins * s + self.min_bin
        o = self.min_freq * self.fratio ** ((p - self.min_bin) / nbins)
        return o


class NegativeAlphaScalingFunctionBins(ScalingFunctionBins):
    @torch.compile
    def forward(self, f: torch.Tensor):
        return super().forward(f) * 2


class NegativeAlphaScalingFunction(ScalingFunction):
    @torch.compile
    def forward(self, f: torch.Tensor):
        return super().forward(f) * 2


class AlphaScalingFunction(nn.Tanh):
    def right_inverse(self, a: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.log((1 + a) / (1 - a))


class C2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.9,
        sr=16000,
        min_freq=None,
        max_freq=None,
        min_bin=None,
        max_bin=None,
    ):
        super().__init__()

        self.sr = sr
        if in_channels != 1:
            self.linear = torch.nn.Linear(in_channels, out_channels, bias=False)

        if combnet.F0_INIT_METHOD == "random":
            self.f = torch.nn.Parameter(
                3 * (torch.rand(out_channels) * 2 - 1),
                requires_grad=True,
            )
        elif combnet.F0_INIT_METHOD == "equal":
            self.f = torch.nn.Parameter(
                torch.linspace(-3, 3, out_channels), requires_grad=True
            )
        elif combnet.F0_INIT_METHOD == "random":
            assert all(p is None for p in [min_freq, max_freq, min_bin, max_bin])
            self.f = torch.nn.Parameter(
                torch.rand(out_channels, in_channels) * (500 - 50) + 50,
                requires_grad=True,
            )
        else:
            raise ValueError(f"unknown initialization method {combnet.F0_INIT_METHOD}")

        if (
            min_freq is not None and max_freq is not None
        ):  # Is this a good way to do this?
            if min_bin is not None and max_bin is not None:
                if isinstance(alpha, float) and alpha < 0:

                    register_parametrization(
                        self,
                        "f",
                        NegativeAlphaScalingFunctionBins(
                            min_freq, max_freq, min_bin, max_bin
                        ),
                    )

                else:

                    register_parametrization(
                        self,
                        "f",
                        ScalingFunctionBins(min_freq, max_freq, min_bin, max_bin),
                    )

            else:
                if isinstance(alpha, float) and alpha < 0:

                    register_parametrization(
                        self,
                        "f",
                        NegativeAlphaScalingFunction(min_freq, max_freq),
                    )
                else:
                    register_parametrization(
                        self, "f", ScalingFunction(min_freq, max_freq)
                    )

        self.a = torch.nn.Parameter(torch.full((out_channels,), alpha))
        register_parametrization(self, "a", AlphaScalingFunction())

    def forward(self, x: Tensor) -> Tensor:
        f = self.f
        num_filters = f.shape[0]
        # print(x.shape, num_filters)

        impulse = x.new_zeros(x.shape[-1])
        impulse[0] = 1.0

        if hasattr(self, "linear"):
            h = self.linear(x.mT).mT  # B x C x T
        else:
            h = x.expand(-1, num_filters, -1)  # B x C x T

        angles = 2 * torch.pi * f / self.sr
        alphas = self.a
        mask = alphas < 0

        radius = alphas.abs() ** (angles * 0.5 / torch.pi)
        full_ir = h.new_empty(num_filters, impulse.shape[0])

        if torch.any(~mask):
            num_pos_alpha = (~mask).count_nonzero().item()
            masked_angles = angles[~mask]
            masked_radius = radius[~mask]

            rp, cp = pos_alpha_even_angle2poles(masked_angles)
            rp, cp = rp * masked_radius, cp * masked_radius
            rp, cp = rp.T, cp.T

            res = poles2res(torch.cat([rp + 0j, cp, cp.conj()], dim=1))
            real_res, cp_res = res[:, :2].real, res[:, 2 : cp.shape[1] + 2]

            # real biquad
            real_b1 = real_res.sum(1)
            real_b2 = -(real_res[:, 0] * rp[:, 1] + real_res[:, 1] * rp[:, 0])
            real_a1 = -rp.sum(1)
            real_a2 = rp.prod(1)

            # conj biquad
            conj_poles_mask = cp.abs() > 0
            filter_indices, poles_indices = torch.nonzero(
                conj_poles_mask, as_tuple=True
            )
            # filter_indices = (
            #     torch.arange(num_pos_alpha, device=cp.device)
            #     .unsqueeze(-1)
            #     .expand(-1, cp.shape[1])[conj_poles_mask]
            # )
            # print(cp.shape, cp_res.shape, conj_poles_mask.shape)
            masked_cp = cp[filter_indices, poles_indices]
            masked_cp_res = cp_res[filter_indices, poles_indices]

            conj_b1 = masked_cp_res.real * 2
            conj_b2 = -2 * (
                masked_cp_res.real * masked_cp.real
                + masked_cp_res.imag * masked_cp.imag
            )
            conj_a1 = -2 * masked_cp.real
            conj_a2 = masked_cp.abs().square()

            num_sections = conj_poles_mask.count_nonzero(dim=1) + 1
            biquad_b = torch.stack(
                [
                    torch.cat(
                        [real_b1 * num_sections, conj_b1 * num_sections[filter_indices]]
                    ),
                    torch.cat(
                        [real_b2 * num_sections, conj_b2 * num_sections[filter_indices]]
                    ),
                ],
                dim=1,
            )  # .repeat(masked_input.shape[0], 1)
            biquad_a = torch.stack(
                [torch.cat([real_a1, conj_a1]), torch.cat([real_a2, conj_a2])], dim=1
            )  # .repeat(masked_input.shape[0], 1)

            B = biquad_b
            A = companion(biquad_a).mT
            # print(A.shape, B.shape)

            # repeated_input = masked_input.repeat_interleave(num_sections, dim=1)
            # print(filter_indices.shape, num_sections, masked_input.shape)
            # aug_input = torch.cat(
            #     [
            #         masked_input,
            #         # masked_input.take_along_dim(filter_indices[None, :, None], dim=1),
            #         masked_input[:, filter_indices, :],
            #     ],
            #     dim=1,
            # )
            ir = state_space(A, impulse.repeat(A.shape[0], 1), B=B, out_idx=0)
            # .unflatten(
            #     0, (aug_input.shape[0], -1)
            # )

            # y = lfilter(biquad_b, biquad_a, aug_input.flatten(0, 1)).unflatten(
            #     0, (aug_input.shape[0], -1)
            # )
            real_ir, conj_ir = ir[:num_pos_alpha], ir[num_pos_alpha:]
            pos_ir = (
                real_ir.index_reduce(
                    0, filter_indices, conj_ir, reduce="mean", include_self=True
                )
                + impulse
            )

            # pos_out = F.conv1d(
            #     masked_input,
            #     pos_ir.unsqueeze(1).flip(-1),
            #     padding=impulse.shape[0] - 1,
            #     groups=masked_input.shape[1],
            # )
            # print(pos_out.shape, x.shape)

            full_ir.masked_scatter_(~mask[:, None], pos_ir)
            # out[:, ~mask] = pos_out
            # out = pos_out

        if torch.any(mask):
            num_neg_alpha = mask.count_nonzero().item()
            masked_angles = angles[mask]
            masked_radius = radius[mask]

            _, cp = neg_alpha_even_angle2poles(masked_angles)
            cp = cp * masked_radius
            cp = cp.T

            res = poles2res(torch.cat([cp, cp.conj()], dim=1))
            cp_res, _ = res.chunk(2, dim=1)

            # conj biquad
            conj_poles_mask = cp.abs() > 0
            filter_indices, poles_indices = torch.nonzero(
                conj_poles_mask, as_tuple=True
            )
            masked_cp = cp[conj_poles_mask]
            masked_cp_res = cp_res[conj_poles_mask]

            conj_b1 = masked_cp_res.real * 2
            conj_b2 = -2 * (
                masked_cp_res.real * masked_cp.real
                + masked_cp_res.imag * masked_cp.imag
            )
            conj_a1 = -2 * masked_cp.real
            conj_a2 = masked_cp.abs().square()

            num_sections = conj_poles_mask.count_nonzero(dim=1)
            biquad_b = (
                torch.stack([conj_b1, conj_b2], dim=1).mul(
                    num_sections[filter_indices].unsqueeze(1)
                )
                # .repeat(masked_input.shape[0], 1)
            )
            biquad_a = torch.stack([conj_a1, conj_a2], dim=1)
            # .repeat(
            # masked_input.shape[0], 1
            # )
            B = biquad_b
            A = companion(biquad_a).mT

            # aug_input = masked_input[:, filter_indices, :]
            y = state_space(A, impulse.repeat(A.shape[0], 1), B=B, out_idx=0)
            # y = lfilter(biquad_b, biquad_a, aug_input.flatten(0, 1)).unflatten(
            #     0, (aug_input.shape[0], -1)
            # )

            neg_ir = (
                y[:num_neg_alpha].index_reduce(
                    0, filter_indices, y, reduce="mean", include_self=False
                )
                + impulse
            )

            full_ir.masked_scatter_(mask[:, None], neg_ir)

        out = torch.fft.irfft(
            torch.fft.rfft(h, n=impulse.shape[0] * 2 - 1)
            * torch.fft.rfft(full_ir, n=impulse.shape[0] * 2 - 1),
            n=impulse.shape[0] * 2 - 1,
        )[..., : impulse.shape[0]]

        return out


class C2V2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        min_freq,
        max_freq,
        alpha=0.9,
        sr=16000,
        decompose_level=1,
    ):
        super().__init__()
        assert out_channels % (decompose_level * 2 - 1) == 0
        assert alpha > 0, "this is the absolute value version"

        num_filters = out_channels // (2 * decompose_level - 1)
        self.sr = sr
        if in_channels != 1:
            self.linear = torch.nn.Linear(in_channels, num_filters, bias=False)

        # self.log_f = torch.nn.Parameter(
        #     torch.rand(
        #         num_filters,
        #     )
        #     * (math.log(max_freq) - math.log(min_freq))
        #     + math.log(min_freq)
        # )
        self.f = torch.nn.Parameter(
            torch.rand(num_filters) * (max_freq - min_freq) + min_freq
        )

        register_parametrization(self, "f", MinMax(min=min_freq, max=max_freq))

        self.alpha = torch.nn.Parameter(torch.full((num_filters,), alpha))
        register_parametrization(self, "alpha", SmoothingCoef())

        self.decompose_level = decompose_level
        if decompose_level > 1:
            self.register_buffer(
                "combine_weights",
                torch.tensor(
                    [
                        0.5**i
                        for i in [decompose_level - 1]
                        + list(range(decompose_level - 1, 0, -1))
                    ]
                ),
            )

    # @property
    # def f(self):
    #     return torch.exp(self.log_f)

    def forward(self, x: Tensor) -> Tensor:
        f = self.f
        num_filters = f.shape[0]

        impulse = x.new_zeros(x.shape[-1])
        impulse[0] = 1.0

        if hasattr(self, "linear"):
            h = self.linear(x.mT).mT  # B x C x T
        else:
            h = x.expand(-1, num_filters, -1)  # B x C x T

        base_angles = 2 * torch.pi * f / self.sr
        alphas = self.alpha
        neg_alpha_angles = []
        for i in range(1, self.decompose_level):
            base_angles = base_angles * 2
            neg_alpha_angles.append(base_angles)

        pos_alpha_angles = base_angles
        radius = alphas ** (pos_alpha_angles * 0.5 / torch.pi)

        rp, cp = pos_alpha_even_angle2poles(pos_alpha_angles)
        rp, cp = rp * radius, cp * radius
        rp, cp = rp.T, cp.T

        res = poles2res(torch.cat([rp + 0j, cp, cp.conj()], dim=1))
        real_res, cp_res = res[:, :2].real, res[:, 2 : cp.shape[1] + 2]

        # real biquad
        real_b1 = real_res.sum(1)
        real_b2 = -(real_res[:, 0] * rp[:, 1] + real_res[:, 1] * rp[:, 0])
        real_a1 = -rp.sum(1)
        real_a2 = rp.prod(1)

        # conj biquad
        conj_poles_mask = cp.abs() > 0
        filter_indices, poles_indices = torch.nonzero(conj_poles_mask, as_tuple=True)
        masked_cp = cp[filter_indices, poles_indices]
        masked_cp_res = cp_res[filter_indices, poles_indices]

        conj_b1 = masked_cp_res.real * 2
        conj_b2 = -2 * (
            masked_cp_res.real * masked_cp.real + masked_cp_res.imag * masked_cp.imag
        )
        conj_a1 = -2 * masked_cp.real
        conj_a2 = masked_cp.abs().square()

        num_sections = conj_poles_mask.count_nonzero(dim=1) + 1
        biquad_b = torch.stack(
            [
                torch.cat(
                    [real_b1 * num_sections, conj_b1 * num_sections[filter_indices]]
                ),
                torch.cat(
                    [real_b2 * num_sections, conj_b2 * num_sections[filter_indices]]
                ),
            ],
            dim=1,
        )
        biquad_a = torch.stack(
            [torch.cat([real_a1, conj_a1]), torch.cat([real_a2, conj_a2])], dim=1
        )

        B = biquad_b
        A = companion(biquad_a).mT
        ir = state_space(A, impulse.repeat(A.shape[0], 1), B=B, out_idx=0)
        real_ir, conj_ir = ir[: alphas.shape[0]], ir[alphas.shape[0] :]
        pos_ir = (
            real_ir.index_reduce(
                0, filter_indices, conj_ir, reduce="mean", include_self=True
            )
            + impulse
        )

        if len(neg_alpha_angles):
            neg_alpha_angles = torch.cat(neg_alpha_angles, dim=0)
            radius = alphas.repeat(self.decompose_level - 1) ** (
                neg_alpha_angles * 0.5 / torch.pi
            )
            _, cp = neg_alpha_even_angle2poles(neg_alpha_angles)
            cp = cp * radius
            cp = cp.T

            res = poles2res(torch.cat([cp, cp.conj()], dim=1))
            cp_res, _ = res.chunk(2, dim=1)

            # conj biquad
            conj_poles_mask = cp.abs() > 0
            filter_indices, poles_indices = torch.nonzero(
                conj_poles_mask, as_tuple=True
            )
            masked_cp = cp[conj_poles_mask]
            masked_cp_res = cp_res[conj_poles_mask]

            conj_b1 = masked_cp_res.real * 2
            conj_b2 = -2 * (
                masked_cp_res.real * masked_cp.real
                + masked_cp_res.imag * masked_cp.imag
            )
            conj_a1 = -2 * masked_cp.real
            conj_a2 = masked_cp.abs().square()

            num_sections = conj_poles_mask.count_nonzero(dim=1)
            biquad_b = torch.stack([conj_b1, conj_b2], dim=1).mul(
                num_sections[filter_indices].unsqueeze(1)
            )
            biquad_a = torch.stack([conj_a1, conj_a2], dim=1)
            B = biquad_b
            A = companion(biquad_a).mT

            y = state_space(A, impulse.repeat(A.shape[0], 1), B=B, out_idx=0)

            neg_ir = (
                y[: neg_alpha_angles.shape[0]].index_reduce(
                    0, filter_indices, y, reduce="mean", include_self=False
                )
                + impulse
            )

            full_ir = torch.cat([neg_ir, pos_ir], dim=0).unflatten(
                0, (self.decompose_level, -1)
            )
        else:
            full_ir = pos_ir

        convolved = torch.fft.irfft(
            torch.fft.rfft(h, n=impulse.shape[0] * 2 - 1).unsqueeze(1)
            * torch.fft.rfft(full_ir, n=impulse.shape[0] * 2 - 1),
            n=impulse.shape[0] * 2 - 1,
        )[..., : impulse.shape[0]]

        if self.decompose_level > 1:
            weighted = convolved.flip(1) * self.combine_weights[:, None, None]
            out = torch.cat(
                [
                    convolved,
                    weighted.cumsum(1)[:, 1:]
                    / self.combine_weights.cumsum(0)[1:, None, None],
                ],
                dim=1,
            ).flatten(1, 2)
        else:
            out = convolved.squeeze(1)

        return out


class Comb1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.9,
        gain=None,
        use_bias=False,
        learn_alpha=False,
        groups=1,
        learn_gain=False,
        sr=16000,
        comb_fn=None,
        min_freq=None,
        max_freq=None,
        min_bin=None,
        max_bin=None,
        n_taps=10,
    ):
        super().__init__()
        self.n_taps = n_taps
        if comb_fn is None:
            self.comb_fn = (
                combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton
            )
        elif isinstance(comb_fn, str):
            comb_fn = getattr(combnet.filters, comb_fn)
            self.comb_fn = comb_fn
        else:
            self.comb_fn = comb_fn
        self.sr = sr
        self.d = (out_channels, in_channels)
        self.la = learn_alpha
        scaling_parameters = [min_freq, max_freq, min_bin, max_bin]
        if (
            min_freq is not None and max_freq is not None
        ):  # Is this a good way to do this?
            if min_bin is not None and max_bin is not None:
                assert True not in [p is None for p in scaling_parameters]
                nbins = max_bin - min_bin
                fratio = max_freq / min_freq
                if isinstance(alpha, float) and alpha < 0:
                    import warnings

                    warnings.warn("using negative alpha scaling function")

                    @torch.compile
                    def scaling_function(
                        f: torch.Tensor,
                    ):  # f = output_channels x input_channels
                        # f = min_freq * (fratio ** ((max_bin * F.sigmoid(f) - min_bin) / nbins))
                        s = F.sigmoid(f)
                        p = nbins * s + min_bin
                        o = min_freq * fratio ** ((p - min_bin) / nbins)
                        return o * 2

                else:

                    @torch.compile
                    def scaling_function(
                        f: torch.Tensor,
                    ):  # f = output_channels x input_channels
                        # f = min_freq * (fratio ** ((max_bin * F.sigmoid(f) - min_bin) / nbins))
                        s = F.sigmoid(f)
                        p = nbins * s + min_bin
                        o = min_freq * fratio ** ((p - min_bin) / nbins)
                        return o

            else:
                assert min_bin is None and max_bin is None
                fratio = max_freq / min_freq

                @torch.compile
                def scaling_function(f: torch.Tensor):
                    s = F.sigmoid(f)
                    o = min_freq * fratio ** ((s))
                    return o

            self.scaling_function = scaling_function
        else:
            self.scaling_function = None

        if self.scaling_function:
            if combnet.F0_INIT_METHOD == "random":
                self.f = torch.nn.Parameter(
                    3 * (torch.rand(out_channels, in_channels) * 2 - 1),
                    requires_grad=True,
                )
            elif combnet.F0_INIT_METHOD == "equal":
                assert in_channels == 1  # TODO generalize?
                self.f = torch.nn.Parameter(
                    torch.linspace(-3, 3, out_channels)[:, None], requires_grad=True
                )
            else:
                raise ValueError(
                    f"unknown initialization method {combnet.F0_INIT_METHOD}"
                )
        else:
            assert combnet.F0_INIT_METHOD == "random"
            self.f = torch.nn.Parameter(
                torch.rand(out_channels, in_channels) * (500 - 50) + 50,
                requires_grad=True,
            )

        if gain is None:
            gain = 1.0
        if learn_gain:
            self.g = torch.nn.Parameter(gain * torch.ones(self.d), requires_grad=True)
        else:
            self.g = gain * torch.ones(self.d)

        if learn_alpha:
            if alpha is not None:
                self.a = torch.nn.Parameter(
                    alpha * torch.ones(self.d), requires_grad=True
                )
            else:
                self.a = torch.nn.Parameter(
                    torch.rand(out_channels, in_channels) * (0.5 - 0.4) + 0.4,
                    requires_grad=True,
                )
        else:
            self.a = alpha * torch.ones(self.d)

        if use_bias:
            self.b = torch.nn.Parameter(torch.zeros((1, out_channels, 1)))
        else:
            self.b = torch.tensor(0)

    def regularization_losses(self):
        regularization = torch.tensor(0.0, device=self.f.device)
        if not hasattr(self, "r") or self.r.device != self.f.device:
            self.r = r.to(self.f.device)
            self.dr = dr.to(self.f.device)
        for i in range(0, self.f.shape[0]):
            for j in range(0, i):
                if i == j:
                    continue
                f1 = self.f[i, 0]
                f2 = self.f[j, 0]
                w1 = self.dr * torch.exp(
                    -((wrap(self.r, -f1 / 2, f1 / 2) / 80) ** 2)
                )  # harmonic bumps for f1
                w2 = self.dr * torch.exp(
                    -((wrap(self.r, -f2 / 2, f2 / 2) / 80) ** 2)
                )  # harmonic bumps for f2
                regularization += torch.dot(w1 / w1.std(), w2 / w2.std())
        return regularization, (self.g.clamp(min=0.01) ** 0.5).sum()
        # return torch.tensor(0.0, device=self.f.device), torch.tensor(0.0, device=self.f.device)

    # def forward(self, x):
    #     return self(x)

    # @torch.compile()
    def forward(self, x):
        d = x.device
        if self.scaling_function:
            f = self.scaling_function(self.f.to(d))
        else:
            f = self.f.to(d)
        # return lc_batch_comb(x, self.f.to(d), self.a.to(d), self.sr, self.g.to(d)) + self.b.to(d)
        # return fractional_comb_fir_multitap(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp_explicit(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # if self.training:
        out = self.comb_fn(x, f, self.a.to(d), self.sr, n_taps=self.n_taps) + self.b.to(
            d
        )
        return out


@lru_cache
def design_fir_highpass(num_taps, cutoff_hz, sample_rate):
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    n = torch.arange(num_taps) - (num_taps - 1) / 2
    h = -torch.sinc(2 * normalized_cutoff * n)
    h[(num_taps - 1) // 2] += 1
    window = torch.hamming_window(num_taps, periodic=False)
    h = h * window
    return h.flip(0)[None, None]


class FusedComb1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.9,
        gain=None,
        use_bias=False,
        learn_alpha=False,
        groups=1,
        learn_gain=False,
        sr=16000,
        comb_fn=None,
        window_size=None,
        reduction="max",
        stride=None,
        last_stride=True,  # include last partial stride
        min_freq=None,
        max_freq=None,
        min_bin=None,
        max_bin=None,
        n_taps=10,
    ):
        self.n_taps = n_taps
        super().__init__()
        assert reduction in ["max", "sum", "mean"]
        self.reduction = reduction
        if comb_fn is None:
            if reduction == "max":
                self.comb_fn = (
                    combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton_fused
                )
            elif reduction == "sum":
                self.comb_fn = (
                    combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton
                )
            elif reduction == "mean":
                self.comb_fn = (
                    combnet.filters.fractional_comb_fir_multitap_lerp_explicit_triton
                )
            # self.comb_fn = combnet.filters.fractional_comb_fir_multitap_lerp_explicit
        elif isinstance(comb_fn, str):
            comb_fn = getattr(combnet.filters, comb_fn)
            self.comb_fn = comb_fn
        else:
            self.comb_fn = comb_fn

        if window_size is None:
            window_size = combnet.WINDOW_SIZE
        if stride is None:
            stride = combnet.HOPSIZE
        self.window_size = window_size
        self.stride = stride

        self.last_stride = last_stride

        self.sr = sr

        self.d = (out_channels, in_channels)
        self.la = learn_alpha

        scaling_parameters = [min_freq, max_freq, min_bin, max_bin]
        if (
            min_freq is not None and max_freq is not None
        ):  # Is this a good way to do this?
            if min_bin is not None and max_bin is not None:
                assert True not in [p is None for p in scaling_parameters]
                nbins = max_bin - min_bin
                fratio = max_freq / min_freq
                if isinstance(alpha, float) and alpha < 0:
                    import warnings

                    warnings.warn("using negative alpha scaling function")

                    @torch.compile
                    def scaling_function(
                        f: torch.Tensor,
                    ):  # f = output_channels x input_channels
                        s = F.sigmoid(f)
                        p = nbins * s + min_bin
                        o = min_freq * fratio ** ((p - min_bin) / nbins)
                        return o * 2

                else:

                    @torch.compile
                    def scaling_function(
                        f: torch.Tensor,
                    ):  # f = output_channels x input_channels
                        s = F.sigmoid(f)
                        p = nbins * s + min_bin
                        o = min_freq * fratio ** ((p - min_bin) / nbins)
                        return o

            else:
                assert min_bin is None and max_bin is None
                fratio = max_freq / min_freq

                @torch.compile
                def scaling_function(f: torch.Tensor):
                    s = F.sigmoid(f)
                    o = min_freq * fratio ** ((s))
                    return o

            self.scaling_function = scaling_function
        else:
            self.scaling_function = None

        if self.scaling_function:
            if combnet.F0_INIT_METHOD == "random":
                self.f = torch.nn.Parameter(
                    3 * (torch.rand(out_channels, in_channels) * 2 - 1),
                    requires_grad=True,
                )
            elif combnet.F0_INIT_METHOD == "equal":
                assert in_channels == 1  # TODO generalize?
                # self.f = torch.nn.Parameter(torch.linspace(-1, 1, out_channels)[:, None], requires_grad=True)
                self.f = torch.nn.Parameter(
                    torch.linspace(-3, 3, out_channels)[:, None], requires_grad=True
                )
            else:
                raise ValueError(
                    f"unknown initialization method {combnet.F0_INIT_METHOD}"
                )
        else:
            assert combnet.F0_INIT_METHOD == "random"
            self.f = torch.nn.Parameter(
                torch.rand(out_channels, in_channels) * (500 - 50) + 50,
                requires_grad=True,
            )

        if gain is None:
            gain = 1.0
        if learn_gain:
            self.g = torch.nn.Parameter(gain * torch.ones(self.d), requires_grad=True)
        else:
            self.g = gain * torch.ones(self.d)

        if learn_alpha:
            if alpha is not None:
                self.a = torch.nn.Parameter(
                    torch.full(self.d, alpha), requires_grad=True
                )
            else:
                self.a = torch.nn.Parameter(
                    torch.rand(out_channels, in_channels) * (0.5 - 0.4) + 0.4,
                    requires_grad=True,
                )
        else:
            # self.a = alpha * torch.ones(self.d)
            self.register_buffer("a", torch.full(self.d, alpha))

        if use_bias:
            self.b = torch.nn.Parameter(torch.zeros((1, out_channels, 1)))
        else:
            # self.b = torch.tensor(0)
            self.register_buffer("b", torch.tensor(0))

    def regularization_losses(self):
        regularization = torch.tensor(0.0, device=self.f.device)
        if not hasattr(self, "r") or self.r.device != self.f.device:
            self.r = r.to(self.f.device)
            self.dr = dr.to(self.f.device)
        for i in range(0, self.f.shape[0]):
            for j in range(0, i):
                if i == j:
                    continue
                f1 = self.f[i, 0]
                f2 = self.f[j, 0]
                w1 = self.dr * torch.exp(
                    -((wrap(self.r, -f1 / 2, f1 / 2) / 80) ** 2)
                )  # harmonic bumps for f1
                w2 = self.dr * torch.exp(
                    -((wrap(self.r, -f2 / 2, f2 / 2) / 80) ** 2)
                )  # harmonic bumps for f2
                regularization += torch.dot(w1 / w1.std(), w2 / w2.std())
        return regularization, (self.g.clamp(min=0.01) ** 0.5).sum()
        # return torch.tensor(0.0, device=self.f.device), torch.tensor(0.0, device=self.f.device)

    # def forward(self, x):
    #     return self(x)

    # @torch.compile()
    def forward(self, x):
        d = x.device
        # return lc_batch_comb(x, self.f.to(d), self.a.to(d), self.sr, self.g.to(d)) + self.b.to(d)
        # return fractional_comb_fir_multitap(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        # return fractional_comb_fir_multitap_lerp_explicit(x, self.f.to(d), self.a.to(d), self.sr) + self.b.to(d)
        if self.scaling_function:
            f = self.scaling_function(self.f)
        else:
            f = self.f
        # if self.training:
        if self.reduction == "max":
            out = (
                self.comb_fn(
                    x,
                    f,
                    self.a,
                    self.sr,
                    self.window_size,
                    self.stride,
                    n_taps=self.n_taps,
                )
                + self.b
            )
        elif self.reduction == "sum":
            y = self.comb_fn(x, f, self.a, self.sr)
            out = F.avg_pool1d(y, self.window_size, self.stride) * self.window_size
        elif self.reduction == "mean":
            y = self.comb_fn(x, f, self.a, self.sr)
            out = F.avg_pool1d(y, self.window_size, self.stride)
        if not self.last_stride:
            out_length = (x.shape[-1] - self.window_size) // self.stride
            out = out[..., :out_length]
        return out


# Comb1dFIIR = partial(Comb1d, comb_fn=combnet.filters.fractional_comb_fiir)


class CombInterference1d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.6,
        gain=None,
        use_bias=False,
        learn_alpha=False,
        groups=1,
        learn_gain=False,
        sr=16000,
    ):
        super().__init__()

        self.sr = sr

        self.d = (out_channels, in_channels)
        self.la = learn_alpha

        self.f = torch.nn.Parameter(
            torch.rand(out_channels, in_channels) * (500 - 50) + 50, requires_grad=True
        )

        if gain is None:
            gain = 1.0
        if learn_gain:
            self.g = torch.nn.Parameter(gain * torch.ones(self.d), requires_grad=True)
        else:
            self.g = gain * torch.ones(self.d)

        if learn_alpha:
            if alpha is not None:
                self.a = torch.nn.Parameter(
                    alpha * torch.ones(self.d), requires_grad=True
                )
            else:
                self.a = torch.nn.Parameter(
                    torch.rand(out_channels, in_channels) * (0.5 - 0.4) + 0.4,
                    requires_grad=True,
                )
        else:
            self.a = alpha * torch.ones(self.d)

        if use_bias:
            self.b = torch.nn.Parameter(torch.zeros((1, out_channels, 1)))
        else:
            self.b = torch.tensor(0)

    def __call__(self, x):
        x = fractional_anticomb_interference_fiir(
            x, self.f, self.a.to(x.device), self.sr
        )
        x = fractional_comb_fiir(x, self.f, self.a.to(x.device), self.sr)  # + self.b
        return x


class CombResidual1d(CombInterference1d):

    def __call__(self, x):
        x = fractional_anticomb_interference_fiir(
            x, self.f, self.a.to(x.device), self.sr, residual_mode=True
        )
        x = fractional_comb_fiir(x, self.f, self.a.to(x.device), self.sr)  # + self.b
        return x
