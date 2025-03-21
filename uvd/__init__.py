from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from .decomp import *
from .models import *


def get_uvd_subgoals(
    frames: np.ndarray | str,
    preprocessor_name: Literal["vip", "r3m", "liv", "clip", "vc1", "dinov2"] = "vip",
    device: torch.device | str | None = "cuda",
    return_indices: bool = False,
    subgoal_target: int | None = None,
    orig_method: bool = False,
    return_intermediate_curves: bool = False) -> list | np.ndarray:
    """Quick API for UVD decomposition."""
    if isinstance(frames, str):
        from decord import VideoReader

        vr = VideoReader(frames, height=224, width=224)
        frames = vr[:].asnumpy()
    preprocessor = get_preprocessor(preprocessor_name, device=device)
    rep = preprocessor.process(frames, return_numpy=True)
    _, decomp_meta = decomp_trajectories("embed", rep, subgoal_target=subgoal_target,
                                         orig_method=orig_method, return_intermediate_curves=return_intermediate_curves)
    indices = decomp_meta.milestone_indices
    if return_indices:
        if return_intermediate_curves:
            return indices, decomp_meta.iter_curves
        return indices
    return frames[indices]
