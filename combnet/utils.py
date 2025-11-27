import torch
from torch import Tensor


def pos_alpha_even_angle2poles(angles: Tensor):
    r"""
    Given an angle in [0, 2 pi), return the poles of an all-pole continuous time comb filter.
    y = x[n] + y[n - 2 pi / angle]

    This function guarantees that when the delay samples are even (e.g., 2, 4, 6, ...)
    it equals to a discrete comb filter.
    """

    min_angle = angles.min()
    max_bins = (torch.pi // min_angle).long().item() + 1

    series = torch.arange(1, max_bins, device=angles.device).unsqueeze(1) * angles
    mask = series < torch.pi

    angles_len = mask.count_nonzero(dim=0) + 1
    target_interval = torch.pi / angles_len
    fadein = series > (torch.pi - target_interval)
    poles = torch.exp(1j * series) * torch.where(
        fadein, (torch.pi - series) / target_interval, 1
    )
    return (
        torch.stack([torch.ones_like(angles), -torch.ones_like(angles)], 0),
        torch.where(mask, poles, 0.0),
    )


def neg_alpha_even_angle2poles(angles: Tensor):
    r"""
    Given an angle in [0, 2 pi), return the poles of an all-pole continuous time comb filter.
    y = x[n] - y[n - 2 pi / angle]

    This function guarantees that when the delay samples are even (e.g., 2, 4, 6, ...)
    it equals to a discrete comb filter.
    """

    min_angle = angles.min()
    max_bins = (torch.pi - min_angle * 0.5) // min_angle + 1

    series = (
        0.5 + torch.arange(0, max_bins.item(), device=angles.device).unsqueeze(1)
    ) * angles
    mask = series < torch.pi

    angles_len = mask.count_nonzero(dim=0)
    target_interval = torch.pi / angles_len * 0.5
    fadein = series > (torch.pi - target_interval)
    poles = torch.exp(1j * series) * torch.where(
        fadein, (torch.pi - series) / target_interval, 1
    )
    return None, torch.where(mask, poles, 0.0)


def pos_alpha_odd_angle2poles(angles: Tensor):
    r"""
    Given an angle in [0, 2 pi), return the poles of an all-pole continuous time comb filter.
    y = x[n] + y[n - 2 pi / angle]

    This function guarantees that when the delay samples are odd (e.g., 1, 3, 5, ...)
    it equals to a discrete comb filter.
    """

    min_angle = angles.min()
    max_bins = torch.pi // min_angle + 1

    series = (
        torch.arange(1, max_bins.item(), device=angles.device).unsqueeze(1) * angles
    )
    mask = series < torch.pi

    angles_len = mask.count_nonzero(dim=0) + 1
    target_interval = torch.pi / (2 * angles_len - 1)
    fadein = series > (torch.pi - target_interval)
    poles = torch.exp(1j * series) * torch.where(
        fadein, (torch.pi - series) / target_interval, 1
    )  # ** 0.5
    return torch.ones_like(angles).unsqueeze(0), torch.where(mask, poles, 0.0)


def neg_alpha_odd_angle2pole(angles: Tensor):
    r"""
    Given an angle in [0, 2 pi), return the poles of an all-pole continuous time comb filter.
    y = x[n] - y[n - 2 pi / angle]

    This function guarantees that when the delay samples are odd (e.g., 1, 3, 5, ...)
    it equals to a discrete comb filter.
    """

    min_angle = angles.min()
    max_bins = (torch.pi - min_angle * 0.5) // min_angle + 1

    series = (
        0.5 + torch.arange(0, max_bins.item(), device=angles.device).unsqueeze(1)
    ) * angles
    mask = series < torch.pi

    angles_len = mask.count_nonzero(dim=0)
    target_interval = 2 * torch.pi / (2 * angles_len + 1)
    fadein = series > (torch.pi - target_interval)
    poles = torch.exp(1j * series) * torch.where(
        fadein, (torch.pi - series) / target_interval, 1
    )
    # return: real poles, complex conjugate poles
    return -torch.ones_like(angles).unsqueeze(0), torch.where(mask, poles, 0.0)


def poles2res(poles: Tensor):
    r"""
    A utility function that returns residual coefficients for
    computing an all-pole filter using parallel one-pole sections.

    H(z) = \sum_i res_i / (1 - p_i z^{-1})
    """
    # method 1: based on partial fraction expansion
    N, M = poles.shape
    mask = poles != 0
    actual_M = mask.count_nonzero(dim=1)
    aug_poles = torch.cat([poles[:, 1:], poles[:, :-1]], dim=1)
    aug_mask = torch.cat([mask[:, 1:], mask[:, :-1]], dim=1)
    windowed_poles = aug_poles.unfold(1, M - 1, 1)
    windowed_mask = aug_mask.unfold(1, M - 1, 1)

    diff = poles.unsqueeze(-1) - windowed_poles
    denom = torch.where(windowed_mask & mask.unsqueeze(-1), diff, 1.0).prod(dim=2)
    res = poles ** (actual_M.unsqueeze(1) - 1) / denom
    return torch.where(mask, res, 0.0)
