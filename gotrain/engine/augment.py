"""Geometric augmentations valid for overhead imagery."""

import torch


def d4(x: torch.Tensor, k: int) -> torch.Tensor:
    """One of the 8 dihedral transforms (k in 0..7) on the last two dims: 90° rotations + flips."""
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if k >= 4 else x


def d4_inverse(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= 4:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -(k % 4), dims=(-2, -1))
