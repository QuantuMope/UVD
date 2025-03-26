from __future__ import annotations

import random
from typing import Literal, NamedTuple, Callable

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from scipy.signal import medfilt
from scipy.signal import savgol_filter, argrelextrema

import uvd.utils as U
from uvd.decomp.kernel_reg import KernelRegression


def linear_random_skip(
    cur_step: int, next_goal_step: int, ratio: float = 0.0, progress_lower: float = 0.3
) -> bool:
    """Progress toward next goal exceed `progress_lower`, linearly increasing
    temperature for P(skip) <= ratio."""
    if ratio == 0.0 or progress_lower == 1:
        return False
    assert cur_step <= next_goal_step, f"{cur_step} > {next_goal_step}"
    # linearly increase until ratio
    progress = cur_step / next_goal_step
    temp = (
        1.0
        if progress < progress_lower
        else ((progress - progress_lower) / (1 - progress_lower)) * ratio
    )
    if 0 < temp < random.random():
        return True
    return False


class DecompMeta(NamedTuple):
    milestone_indices: list
    milestone_starts: list | None = None
    iter_curves: list[np.ndarray] | None = None


def _debug_plt(
    xs: np.ndarray,
    embed_distances: np.ndarray,
    starts: np.ndarray | list,
    ends: np.ndarray | list,
    return_numpy: bool = False,
):
    fig = plt.figure()
    for s in starts:
        plt.axvline(x=s, linestyle="--")
    for e in ends:
        plt.axvline(x=e, linestyle="dotted")
    plt.plot(xs, np.gradient(embed_distances), linewidth=1.5, label="1st derivative")
    plt.plot(
        xs,
        np.gradient(np.gradient(embed_distances)),
        linewidth=1.5,
        label="2nd derivative",
    )
    plt.plot(xs, embed_distances, linewidth=1.5, label="embedding distance")
    plt.legend()
    if return_numpy:
        # fig.canvas.draw()
        return U.plt_to_numpy(fig)
    else:
        plt.show()


def embed_decomp_no_robot(
    embeddings: np.ndarray | torch.Tensor,
    no_robot_embeddings: np.ndarray | torch.Tensor,
    window_length: int = 10,
    derivative_order: int = 1,
    derivative_threshold: float = 1e-3,
    threshold_subgoal_passing: float | None = None,
    force_interleave: bool = False,
    debug_plt: bool = False,
    debug_plt_to_wandb: bool = False,
    task_name: str | None = None,
    fill_embeddings: bool = True,
):
    debug_plt = debug_plt and U.is_rank_zero()
    debug_plt_to_wandb = debug_plt and debug_plt_to_wandb
    if threshold_subgoal_passing is not None:
        assert 0 < threshold_subgoal_passing <= 1.0, threshold_subgoal_passing
    if isinstance(embeddings, torch.Tensor):
        device = embeddings.device
    else:
        device = None
    # clip based preprocessors would have bf16 by default
    embeddings = U.any_to_numpy(embeddings, dtype="float32")
    no_robot_embeddings = U.any_to_numpy(no_robot_embeddings, dtype="float32")
    # L, N (though can be rgb for debugging as well)
    traj_length = embeddings.shape[0]

    debug_plt_figs = []
    if fill_embeddings:
        milestone_embeddings = []
    else:
        milestone_embeddings = None
    # milestone indices, i.e. end of obj changing
    subgoal_indices = []
    # indices when obj changing starts, i.e. milestone indices for hand reaching
    subgoal_starts = []
    cur_subgoal_idx = traj_length - 1
    # back to start
    while cur_subgoal_idx > 15:
        unnormalized_embed_distance = np.linalg.norm(
            no_robot_embeddings[: cur_subgoal_idx + 1]
            - no_robot_embeddings[cur_subgoal_idx],
            axis=1,
        )
        cur_embed_distance = unnormalized_embed_distance / np.linalg.norm(
            no_robot_embeddings[0] - no_robot_embeddings[cur_subgoal_idx]
        )
        cur_embed_distance = medfilt(cur_embed_distance, kernel_size=None)  # smooth

        if derivative_order == 1:
            slope = np.gradient(cur_embed_distance)
        elif derivative_order == 2:
            slope = np.gradient(np.gradient(cur_embed_distance))
        else:
            raise NotImplementedError(derivative_order)
        valid_slope_indices = np.where(np.abs(slope) >= derivative_threshold)[0]
        # Find the differences between consecutive valid_slope_indices
        diffs = np.diff(valid_slope_indices)
        # Find indices where the difference is greater than window_length + 1
        break_indices = np.where(diffs > window_length + 1)[0]
        # Extract start and end positions
        start_positions = valid_slope_indices[np.concatenate([[0], break_indices + 1])]
        end_positions = valid_slope_indices[
            np.concatenate((break_indices, [len(valid_slope_indices) - 1]))
        ]

        # small tolerance
        start_positions = [max(0, p - 1) for p in start_positions]
        end_positions = [min(traj_length - 1, p + 1) for p in end_positions]

        if debug_plt:
            x = np.arange(0, len(cur_embed_distance))
            fig = _debug_plt(
                x,
                cur_embed_distance,
                start_positions,
                end_positions,
                return_numpy=debug_plt_to_wandb,
            )
            if debug_plt_to_wandb:
                debug_plt_figs.append(fig)

        subgoal_indices.append(cur_subgoal_idx)
        subgoal_starts.append(start_positions[-1])
        if len(end_positions) < 2 or end_positions[-2] < 15:
            if threshold_subgoal_passing is not None:
                cur_milestone_dist = None
                if len(subgoal_indices) > 1:
                    cur_milestone_dist = np.linalg.norm(
                        no_robot_embeddings[cur_subgoal_idx]  # cur
                        - no_robot_embeddings[subgoal_indices[-2]]  # prev
                    )  # use the order from start to end
                if fill_embeddings:
                    for step in reversed(range(cur_subgoal_idx + 1)):
                        if (
                            len(subgoal_indices) == 1
                            or unnormalized_embed_distance[step] / cur_milestone_dist
                            > threshold_subgoal_passing
                        ):
                            milestone_embeddings.append(embeddings[cur_subgoal_idx])
                        else:
                            milestone_embeddings.append(embeddings[subgoal_indices[-2]])
            break
        if threshold_subgoal_passing is not None:
            cur_milestone_dist = None
            if len(subgoal_indices) > 1:
                cur_milestone_dist = np.linalg.norm(
                    no_robot_embeddings[cur_subgoal_idx]  # cur
                    - no_robot_embeddings[subgoal_indices[-2]]  # prev
                )
            if fill_embeddings:
                for step in reversed(range(end_positions[-2] + 1, cur_subgoal_idx + 1)):
                    # not pass threshold or 1st iter with only last frame contained
                    if (
                        len(subgoal_indices) == 1
                        or unnormalized_embed_distance[step] / cur_milestone_dist
                        > threshold_subgoal_passing
                    ):
                        # use the real subgoal
                        milestone_embeddings.append(embeddings[cur_subgoal_idx])
                    else:
                        # skip to the next subgoal if passing threshold
                        milestone_embeddings.append(embeddings[subgoal_indices[-2]])
        cur_subgoal_idx = end_positions[-2]

    subgoal_starts = list(reversed(subgoal_starts))
    subgoal_indices = list(reversed(subgoal_indices))

    if len(debug_plt_figs) > 0 and wandb.run is not None:
        debug_plt_figs = np.concatenate(debug_plt_figs, axis=1)
        wandb.log(
            {
                f"decomp_curves/{task_name}": wandb.Image(
                    debug_plt_figs,
                    caption=f"starts: {subgoal_starts}, ends: {subgoal_indices}",
                )
            }
        )

    assert len(subgoal_starts) == len(
        subgoal_indices
    ), f"{subgoal_starts}, {subgoal_indices}"

    if force_interleave:
        _starts, _ends = [], []
        for i in range(len(subgoal_starts)):
            if subgoal_starts[i] < subgoal_indices[i]:
                _starts.append(subgoal_starts[i])
                _ends.append(subgoal_indices[i])
            else:
                U.get_logger().warning(
                    f"{subgoal_starts} & {subgoal_indices} not interleaved"
                )

    if fill_embeddings:
        if threshold_subgoal_passing is not None:
            milestone_embeddings = np.stack(list(reversed(milestone_embeddings)))
        else:
            # slightly faster to do once here without threshold checking
            milestone_embeddings = np.concatenate(
                [embeddings[subgoal_indices[0], ...][None]]
                + [
                    np.full((end - start, *embeddings.shape[1:]), embeddings[end, ...])
                    for start, end in zip([0] + subgoal_indices[:-1], subgoal_indices)
                ],
            )

        if device is not None:
            milestone_embeddings = U.any_to_torch_tensor(
                milestone_embeddings, device=device
            )
    return milestone_embeddings, DecompMeta(
        milestone_indices=subgoal_indices, milestone_starts=subgoal_starts
    )


def embed_decomp_no_robot_extended(
    embeddings: np.ndarray | torch.Tensor,
    no_robot_embeddings: np.ndarray | torch.Tensor,
    threshold_subgoal_passing: float | None = None,
    **kwargs,
):
    kwargs["fill_embeddings"] = False
    _, decomp_meta = embed_decomp_no_robot(
        embeddings,
        no_robot_embeddings,
        threshold_subgoal_passing=None,
        **kwargs,
    )
    milestone_indices = decomp_meta.milestone_indices
    milestone_starts = decomp_meta.milestone_starts
    norm = (
        np.linalg.norm if isinstance(embeddings[0], np.ndarray) else torch.linalg.norm
    )
    assert len(milestone_starts) == len(milestone_indices)

    milestone_embeddings = []
    hybrid_indices = list(sorted(milestone_starts + milestone_indices))
    prev_idx = -1
    s = -1
    init_dist = None
    for i, goal_idx in enumerate(hybrid_indices):
        once_passed = False
        for _ in range(goal_idx - prev_idx):
            s += 1
            if threshold_subgoal_passing is None:
                milestone_embeddings.append(embeddings[goal_idx])
            elif once_passed:
                milestone_embeddings.append(embeddings[hybrid_indices[i + 1]])
            else:
                raw_cur_dist = float(norm(embeddings[s] - embeddings[goal_idx]))
                if init_dist is None:
                    assert s == 0, s
                    cur_dist = 1.0
                    init_dist = max(raw_cur_dist, 1e-7)
                else:
                    cur_dist = raw_cur_dist / init_dist
                if (
                    cur_dist <= threshold_subgoal_passing
                    and i < len(hybrid_indices) - 1
                ):
                    init_dist = float(
                        norm(embeddings[s] - embeddings[hybrid_indices[i + 1]])
                    )
                    init_dist = max(init_dist, 1e-7)
                    once_passed = True
                    milestone_embeddings.append(embeddings[hybrid_indices[i + 1]])
                else:
                    milestone_embeddings.append(embeddings[goal_idx])
        prev_idx = goal_idx

    milestone_embeddings = U.any_stack(milestone_embeddings)
    U.assert_(milestone_embeddings.shape, embeddings.shape)
    return milestone_embeddings, DecompMeta(milestone_indices=hybrid_indices)


def get_hybrid_milestones(
    start_embeddings: np.ndarray,  # w. robot
    end_embeddings: np.ndarray,  # w.o robot
    milestone_starts: list,
    milestone_indices: list,
) -> np.ndarray:
    assert len(milestone_starts) == len(milestone_indices)
    assert len(start_embeddings) == len(end_embeddings)
    milestone_only = len(milestone_starts) == len(start_embeddings)
    hybrid_milestones = np.empty(
        (start_embeddings.shape[0] * 2, *start_embeddings.shape[1:]),
        dtype=start_embeddings.dtype,
    )
    hybrid_milestones[::2] = (
        start_embeddings if milestone_only else start_embeddings[milestone_starts]
    )
    hybrid_milestones[1::2] = (
        end_embeddings if milestone_only else end_embeddings[milestone_indices]
    )
    return hybrid_milestones


# def embedding_decomp(
#     embeddings: np.ndarray | torch.Tensor,
#     normalize_curve: bool = True,
#     min_interval: int = 18,
#     window_length: int | None = None,
#     smooth_method: Literal["kernel", "savgol"] = "kernel",
#     extrema_comparator: Callable = np.greater,
#     fill_embeddings: bool = True,
#     return_intermediate_curves: bool = False,
#     **kwargs,
# ) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
#     if torch.is_tensor(embeddings):
#         device = embeddings.device
#         embeddings = U.any_to_numpy(embeddings)
#     else:
#         device = None
#     # L, N
#     assert embeddings.ndim == 2, embeddings.shape
#     traj_length = embeddings.shape[0]
#
#     cur_goal_idx = traj_length - 1
#     goal_indices = [cur_goal_idx]
#     cur_embeddings = embeddings[
#         max(0, cur_goal_idx - (window_length or cur_goal_idx)) : cur_goal_idx + 1
#     ]
#     iterate_num = 0
#     iter_curves = [] if return_intermediate_curves else None
#     while cur_goal_idx > (window_length or min_interval):
#         iterate_num += 1
#         # get goal embedding
#         goal_embedding = cur_embeddings[-1]
#         distances = np.linalg.norm(cur_embeddings - goal_embedding, axis=1)
#         if normalize_curve:
#             distances = distances / np.linalg.norm(cur_embeddings[0] - goal_embedding)
#
#         x = np.arange(
#             max(0, cur_goal_idx - (window_length or cur_goal_idx)), cur_goal_idx + 1
#         )
#
#         if smooth_method == "kernel":
#             smooth_kwargs = dict(kernel="rbf", gamma=0.08)
#             smooth_kwargs.update(kwargs or {})
#             kr = KernelRegression(**smooth_kwargs)
#             kr.fit(x.reshape(-1, 1), distances)
#             distance_smoothed = kr.predict(x.reshape(-1, 1))
#         elif smooth_method == "savgol":
#             smooth_kwargs = dict(window_length=85, polyorder=2, mode="nearest")
#             smooth_kwargs.update(kwargs or {})
#             distance_smoothed = savgol_filter(distances, **smooth_kwargs)
#         elif smooth_method is None:
#             distance_smoothed = distances
#         else:
#             raise NotImplementedError(smooth_method)
#
#         if iter_curves is not None:
#             iter_curves.append(distance_smoothed)
#
#         extrema_indices = argrelextrema(distance_smoothed, extrema_comparator)[0]
#         x_extrema = x[extrema_indices]
#
#         update_goal = False
#         for i in range(len(x_extrema) - 1, -1, -1):
#             if cur_goal_idx < min_interval:
#                 break
#             if (
#                 cur_goal_idx - x_extrema[i] > min_interval
#                 and x_extrema[i] > min_interval
#             ):
#                 cur_goal_idx = x_extrema[i]
#                 update_goal = True
#                 goal_indices.append(cur_goal_idx)
#                 break
#
#         if not update_goal or cur_goal_idx < min_interval:
#             break
#         cur_embeddings = embeddings[
#             max(0, cur_goal_idx - (window_length or cur_goal_idx)) : cur_goal_idx + 1
#         ]
#
#     goal_indices = goal_indices[::-1]
#     if fill_embeddings:
#         milestone_embeddings = np.concatenate(
#             [embeddings[goal_indices[0], ...][None]]
#             + [
#                 np.full((end - start, *embeddings.shape[1:]), embeddings[end, ...])
#                 for start, end in zip([0] + goal_indices[:-1], goal_indices)
#             ],
#         )
#         if device is not None:
#             milestone_embeddings = U.any_to_torch_tensor(
#                 milestone_embeddings, device=device
#             )
#     else:
#         milestone_embeddings = None
#     return milestone_embeddings, DecompMeta(
#         milestone_indices=goal_indices, iter_curves=iter_curves
#     )

def first_derivative(curve):
    """
    Calculate the first derivative of a curve using central differences.

    Args:
        curve: Array of y-values of the curve

    Returns:
        Array of first derivatives (same length as input)
    """
    # Initialize derivative array with same length as input
    derivative = np.zeros_like(curve)

    # Use central differences for interior points
    for i in range(1, len(curve) - 1):
        derivative[i] = (curve[i + 1] - curve[i - 1]) / 2

    # Forward difference for first point
    derivative[0] = curve[1] - curve[0]

    # Backward difference for last point
    derivative[-1] = curve[-1] - curve[-2]

    return derivative

def embedding_decomp(
        embeddings: np.ndarray | torch.Tensor,
        normalize_curve: bool = True,
        min_interval: int = 18,
        window_length: int | None = None,
        smooth_method: Literal["kernel", "savgol"] = "kernel",
        extrema_comparator: Callable = np.greater,
        fill_embeddings: bool = True,
        return_intermediate_curves: bool = False,
        subgoal_target: int | None = None,
        orig_method: bool = False,
        **kwargs):
    if orig_method:
        return embedding_decomp_orig(
            embeddings, normalize_curve, min_interval, window_length, smooth_method,
            extrema_comparator, fill_embeddings, return_intermediate_curves, subgoal_target,
            **kwargs
        )
    else:
        return embedding_decomp_custom(
            embeddings, normalize_curve, min_interval, window_length, smooth_method,
            extrema_comparator, fill_embeddings, return_intermediate_curves, subgoal_target,
            **kwargs)

def embedding_decomp_orig(
        embeddings: np.ndarray | torch.Tensor,
        normalize_curve: bool = True,
        min_interval: int = 18,
        window_length: int | None = None,
        smooth_method: Literal["kernel", "savgol"] = "kernel",
        extrema_comparator: Callable = np.greater,
        fill_embeddings: bool = True,
        return_intermediate_curves: bool = False,
        subgoal_target: int | None = None,
        **kwargs,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    if torch.is_tensor(embeddings):
        device = embeddings.device
        embeddings = U.any_to_numpy(embeddings)
    else:
        device = None
    # L, N
    assert embeddings.ndim == 2, embeddings.shape
    traj_length = embeddings.shape[0]
    cur_goal_idx = traj_length - 1
    goal_indices = [cur_goal_idx]
    cur_embeddings = embeddings[
                     max(0, cur_goal_idx - (window_length or cur_goal_idx)): cur_goal_idx + 1
                     ]
    iterate_num = 0
    iter_curves = [] if return_intermediate_curves else None
    # Dictionary to store extrema sharpness for later filtering
    extrema_sharpness = {}


    while cur_goal_idx > (window_length or min_interval):
        iterate_num += 1
        # get goal embedding
        goal_embedding = cur_embeddings[-1]
        distances = np.linalg.norm(cur_embeddings - goal_embedding, axis=1)
        if normalize_curve:
            distances = distances / np.linalg.norm(cur_embeddings[0] - goal_embedding)
        x = np.arange(
            max(0, cur_goal_idx - (window_length or cur_goal_idx)), cur_goal_idx + 1
        )
        if smooth_method == "kernel":
            smooth_kwargs = dict(kernel="rbf", gamma=0.08)
            smooth_kwargs.update(kwargs or {})
            kr = KernelRegression(**smooth_kwargs)
            kr.fit(x.reshape(-1, 1), distances)
            distance_smoothed = kr.predict(x.reshape(-1, 1))
        elif smooth_method == "savgol":
            smooth_kwargs = dict(window_length=85, polyorder=2, mode="nearest")
            smooth_kwargs.update(kwargs or {})
            distance_smoothed = savgol_filter(distances, **smooth_kwargs)
        elif smooth_method is None:
            distance_smoothed = distances
        else:
            raise NotImplementedError(smooth_method)
        if iter_curves is not None:
            iter_curves.append(distance_smoothed)

        extrema_indices = argrelextrema(distance_smoothed, np.greater)[0]

        # maxima_indices = argrelextrema(distance_smoothed, np.greater)[0]
        # minima_indices = argrelextrema(distance_smoothed, np.less)[0]
        # extrema_indices = np.union1d(maxima_indices, minima_indices)
        #
        # extrema_indices = merge_nearby_extrema(extrema_indices, distance_threshold=15)

        # plt.plot(np.arange(len(distance_smoothed)), distance_smoothed)
        # for idx in extrema_indices:
        #     plt.axvline(x=idx, color='r')
        # plt.show()
        # exit(0)

        # Calculate sharpness for each extrema
        extrema_indices = list(extrema_indices)
        x = list(x)
        for idx in extrema_indices[:]:

            from_stationary = check_if_adjacent_to_stationary(distance_smoothed, idx, extrema_indices)
            if from_stationary:
                extrema_indices.remove(idx)
                del x[idx]
                continue

            actual_idx = x[idx]
            y_dist = calculate_y_distance(distance_smoothed, idx, extrema_indices)
            extrema_sharpness[actual_idx] = y_dist
            print(y_dist)

        extrema_indices = np.array(extrema_indices)
        x = np.array(x)

        if len(extrema_indices) == 0:
            break

        # Calculate sharpness for each extrema using a combined approach
        x_extrema = x[extrema_indices]
        update_goal = False
        for i in range(len(x_extrema) - 1, -1, -1):
            if cur_goal_idx < min_interval:
                break
            if (
                    cur_goal_idx - x_extrema[i] > min_interval
                    and x_extrema[i] > min_interval
            ):
                cur_goal_idx = x_extrema[i]
                update_goal = True
                goal_indices.append(cur_goal_idx)
                break
        if not update_goal or cur_goal_idx < min_interval:
            break
        cur_embeddings = embeddings[
                         max(0, cur_goal_idx - (window_length or cur_goal_idx)): cur_goal_idx + 1
                         ]

    goal_indices = goal_indices[::-1]
    # for i in goal_indices[:-1]:
    #     print(extrema_sharpness[i])

    # Filter indices based on subgoal_target if provided
    if subgoal_target is not None and len(goal_indices) > subgoal_target:
        # Always keep the last index (end point)
        if subgoal_target >= 1:
            # Get all indices except the last one, with their sharpness
            candidate_indices = goal_indices[:-1]
            indices_with_sharpness = [
                (idx, extrema_sharpness.get(idx, 0)) for idx in candidate_indices
            ]

            # Sort by sharpness (higher values first)
            sorted_indices = [
                idx for idx, _ in sorted(
                    indices_with_sharpness,
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

            # Take top (subgoal_target-1) indices by sharpness
            top_indices = sorted_indices[:subgoal_target - 1]

            # Add the last index and sort to maintain chronological order
            filtered_indices = top_indices + [goal_indices[-1]]
            goal_indices = sorted(filtered_indices)
        else:
            # If subgoal_target is 0, just keep the last index
            goal_indices = [goal_indices[-1]]
    if fill_embeddings:
        milestone_embeddings = np.concatenate(
            [embeddings[goal_indices[0], ...][None]]
            + [
                np.full((end - start, *embeddings.shape[1:]), embeddings[end, ...])
                for start, end in zip([0] + goal_indices[:-1], goal_indices)
            ],
        )
        if device is not None:
            milestone_embeddings = U.any_to_torch_tensor(
                milestone_embeddings, device=device
            )
    else:
        milestone_embeddings = None

    return milestone_embeddings, DecompMeta(
        milestone_indices=goal_indices, iter_curves=iter_curves
    )


def calculate_prominence(curve, idx, is_maxima):
    """Calculate prominence with better boundary handling"""
    val = curve[idx]

    # Skip extrema that are at the very edges
    if idx == 0 or idx == len(curve) - 1:
        return 0

    if is_maxima:  # For peaks
        # Find key points: all higher peaks to left and right
        higher_left = [i for i in range(0, idx) if curve[i] > val]
        higher_right = [i for i in range(idx + 1, len(curve)) if curve[i] > val]

        # If no higher peaks on either side, use curve endpoints
        left_ref = max(higher_left) if higher_left else 0
        right_ref = min(higher_right) if higher_right else len(curve) - 1

        # Find lowest point between peak and references
        left_min = np.min(curve[left_ref:idx + 1])
        right_min = np.min(curve[idx:right_ref + 1])

        # Prominence is minimum height above saddle points
        return val - max(left_min, right_min)
    else:  # For valleys
        # Find key points: all lower valleys to left and right
        lower_left = [i for i in range(0, idx) if curve[i] < val]
        lower_right = [i for i in range(idx + 1, len(curve)) if curve[i] < val]

        # If no lower valleys on either side, use curve endpoints
        left_ref = max(lower_left) if lower_left else 0
        right_ref = min(lower_right) if lower_right else len(curve) - 1

        # Find highest point between valley and references
        left_max = np.max(curve[left_ref:idx + 1])
        right_max = np.max(curve[idx:right_ref + 1])

        # Prominence is minimum depth below saddle points
        return min(left_max, right_max) - val


def check_if_adjacent_to_stationary(curve, idx, extrema_indices,
                                    threshold: float = 0.00):
    """
    Calculate the significance of an extrema based on y-distance to neighboring extrema.
    For edge extrema, uses the curve boundaries.

    Args:
        curve: The smoothed distance curve (y values)
        idx: Index of the extrema in the curve
        extrema_indices: Array of all extrema indices

    Returns:
        A value representing the significance based on y-distance change
    """
    val = curve[idx]
    curve_length = len(curve)

    # Find the indices of neighboring extrema
    extrema_positions = np.where(extrema_indices == idx)[0][0]  # Find position in extrema_indices array

    # Find left neighbor (previous extrema or curve start)
    if extrema_positions > 0:
        left_extrema_idx = extrema_indices[extrema_positions - 1]
        left_val = curve[left_extrema_idx]
    else:
        # If this is the leftmost extrema, use the start of the curve
        left_extrema_idx = 0
        left_val = curve[0]

    # Find right neighbor (next extrema or curve end)
    if extrema_positions < len(extrema_indices) - 1:
        right_extrema_idx = extrema_indices[extrema_positions + 1]
        right_val = curve[right_extrema_idx]
    else:
        # If this is the rightmost extrema, use the end of the curve
        right_extrema_idx = curve_length - 1
        right_val = curve[-1]

    # Calculate y-distance changes
    # left_distance = abs(val - left_val) / (idx - left_extrema_idx)
    # right_distance = abs(val - right_val) / (right_extrema_idx - idx)
    left_distance = abs(val - left_val)
    right_distance = abs(val - right_val)

    return left_distance < threshold or right_distance < threshold


def calculate_y_distance(curve, idx, extrema_indices):
    """
    Calculate the significance of an extrema based on y-distance to neighboring extrema.
    For edge extrema, uses the curve boundaries.

    Args:
        curve: The smoothed distance curve (y values)
        idx: Index of the extrema in the curve
        extrema_indices: Array of all extrema indices

    Returns:
        A value representing the significance based on y-distance change
    """
    val = curve[idx]
    curve_length = len(curve)

    # Find the indices of neighboring extrema
    extrema_positions = np.where(extrema_indices == idx)[0][0]  # Find position in extrema_indices array

    # Find left neighbor (previous extrema or curve start)
    if extrema_positions > 0:
        left_extrema_idx = extrema_indices[extrema_positions - 1]
        left_val = curve[left_extrema_idx]
    else:
        # If this is the leftmost extrema, use the start of the curve
        left_extrema_idx = 0
        left_val = curve[0]

    # Find right neighbor (next extrema or curve end)
    if extrema_positions < len(extrema_indices) - 1:
        right_extrema_idx = extrema_indices[extrema_positions + 1]
        right_val = curve[right_extrema_idx]
    else:
        # If this is the rightmost extrema, use the end of the curve
        right_extrema_idx = curve_length - 1
        right_val = curve[-1]

    # Calculate y-distance changes
    left_distance = abs(val - left_val)
    right_distance = abs(val - right_val)
    # left_slope = abs((val - left_val) / (idx - left_extrema_idx ))
    # right_slope = abs((val - right_val) / (right_extrema_idx - idx))

    # Combine distances - you can adjust this formula based on your preference
    # Here we take the average of both distances
    # y_distance_score = (left_distance + right_distance) / 2
    if left_distance < 0.05 or right_distance < 0.05:
        y_distance_score = 0
    else:
        # y_distance_score = min(left_distance, right_distance)
        y_distance_score = left_distance + right_distance
    y_distance_score = left_distance + right_distance
    # y_distance_score = min(left_distance, right_distance)

    # Alternative: use the minimum distance to be conservative
    # y_distance_score = min(left_distance, right_distance)

    # Another alternative: use the maximum distance to emphasize large changes
    # y_distance_score = max(left_distance, right_distance)

    return y_distance_score
    # return y_slope_score


def merge_nearby_extrema(extrema_indices, distance_threshold=5):
    """
    Merge extrema indices that are within a certain distance threshold.
    """
    if len(extrema_indices) <= 1:
        return extrema_indices

    # Sort the indices first
    sorted_indices = np.sort(extrema_indices)

    # List to store merged indices
    merged = []
    i = 0

    while i < len(sorted_indices):
        # Add current index
        current = sorted_indices[i]
        merged.append(current)

        # Skip any following indices that are too close
        while i + 1 < len(sorted_indices) and sorted_indices[i + 1] - current <= distance_threshold:
            i += 1

        i += 1

    return np.array(merged)


def embedding_decomp_custom(
        embeddings: np.ndarray | torch.Tensor,
        normalize_curve: bool = True,
        min_interval: int = 18,
        window_length: int | None = None,
        smooth_method: Literal["kernel", "savgol"] = "kernel",
        extrema_comparator: Callable = np.greater,
        fill_embeddings: bool = True,
        return_intermediate_curves: bool = False,
        subgoal_target: int | None = None,
        **kwargs,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    if torch.is_tensor(embeddings):
        device = embeddings.device
        embeddings = U.any_to_numpy(embeddings)
    else:
        device = None
    # L, N
    assert embeddings.ndim == 2, embeddings.shape
    traj_length = embeddings.shape[0]
    cur_goal_idx = traj_length - 1
    goal_indices = [cur_goal_idx]
    cur_embeddings = embeddings[
                     max(0, cur_goal_idx - (window_length or cur_goal_idx)): cur_goal_idx + 1
                     ]
    iterate_num = 0
    iter_curves = [] if return_intermediate_curves else None
    # Dictionary to store extrema sharpness for later filtering
    extrema_sharpness = {}

    while cur_goal_idx > (window_length or min_interval):
        iterate_num += 1
        # get goal embedding
        goal_embedding = cur_embeddings[-1]
        distances = np.linalg.norm(cur_embeddings - goal_embedding, axis=1)
        if normalize_curve:
            distances = distances / np.linalg.norm(cur_embeddings[0] - goal_embedding)
        x = np.arange(
            max(0, cur_goal_idx - (window_length or cur_goal_idx)), cur_goal_idx + 1
        )
        if smooth_method == "kernel":
            smooth_kwargs = dict(kernel="rbf", gamma=0.08)
            smooth_kwargs.update(kwargs or {})
            kr = KernelRegression(**smooth_kwargs)
            kr.fit(x.reshape(-1, 1), distances)
            distance_smoothed = kr.predict(x.reshape(-1, 1))
        elif smooth_method == "savgol":
            smooth_kwargs = dict(window_length=85, polyorder=2, mode="nearest")
            smooth_kwargs.update(kwargs or {})
            distance_smoothed = savgol_filter(distances, **smooth_kwargs)
        elif smooth_method is None:
            distance_smoothed = distances
        else:
            raise NotImplementedError(smooth_method)
        if iter_curves is not None:
            iter_curves.append(distance_smoothed)

        # extrema_indices = argrelextrema(distance_smoothed, extrema_comparator)[0]

        maxima_indices = argrelextrema(distance_smoothed, np.greater)[0]
        minima_indices = argrelextrema(distance_smoothed, np.less)[0]
        extrema_indices = np.union1d(maxima_indices, minima_indices)
        # extrema_indices = merge_nearby_extrema(extrema_indices, distance_threshold=5)

        # plt.plot(np.arange(len(distance_smoothed)), distance_smoothed)
        # for idx in maxima_indices:
        #     plt.axvline(x=idx, color='r')
        # for idx in minima_indices:
        #     plt.axvline(x=idx, color='g')
        # plt.show()

        def better_second_deriv(curve, idx, window=3):
            """Calculate second derivative using a wider window"""
            h = 1  # Assuming uniform spacing
            weights = np.array([1, -2, 1])  # Basic second derivative kernel

            # For wider windows, we need to extend this kernel
            if window > 1:
                # Create a wider kernel for smoother approximation
                extended = np.zeros(2 * window + 1)
                extended[window - 1:window + 2] = weights
                weights = extended

            # Apply the kernel around idx
            start = max(0, idx - window)
            end = min(len(curve), idx + window + 1)
            kernel_start = max(0, window - idx)
            kernel_end = kernel_start + (end - start)

            # Compute the weighted sum
            return np.sum(curve[start:end] * weights[kernel_start:kernel_end]) / (h * h)

        # Calculate sharpness for each extrema
        for idx in extrema_indices:
            actual_idx = x[idx]
            # Calculate sharpness using second derivative magnitude

            # is_maxima = idx in maxima_indices
            # prom = calculate_prominence(distance_smoothed, idx, is_maxima)
            # extrema_sharpness[actual_idx] = prom

            y_dist = calculate_y_distance(distance_smoothed, idx, extrema_indices)
            extrema_sharpness[actual_idx] = y_dist

            # if idx > 0 and idx < len(distance_smoothed) - 1:
            #     second_deriv = better_second_deriv(distance_smoothed, idx, window=5)
            #     # Second derivative approximation
            #     # second_deriv = (
            #     #         distance_smoothed[idx + 1]
            #     #         - 2 * distance_smoothed[idx]
            #     #         + distance_smoothed[idx - 1]
            #     # )
            #     sharpness = abs(second_deriv)
            #     # Store the sharpness value with the actual index
            #     extrema_sharpness[actual_idx] = sharpness
            #     print(sharpness)

        # goal_indices = extrema_indices[::-1]

        goal_indices = list(extrema_indices)
        goal_indices.append(len(x)-1)
        goal_indices = np.array(goal_indices, dtype=int)[::-1]
        extrema_sharpness[len(x)-1] = 0
        break

        # Calculate sharpness for each extrema using a combined approach
        x_extrema = x[extrema_indices]
        update_goal = False
        for i in range(len(x_extrema) - 1, -1, -1):
            if cur_goal_idx < min_interval:
                break
            if (
                    cur_goal_idx - x_extrema[i] > min_interval
                    and x_extrema[i] > min_interval
            ):
                cur_goal_idx = x_extrema[i]
                update_goal = True
                goal_indices.append(cur_goal_idx)
                break
        if not update_goal or cur_goal_idx < min_interval:
            break
        cur_embeddings = embeddings[
                         max(0, cur_goal_idx - (window_length or cur_goal_idx)): cur_goal_idx + 1
                         ]

    goal_indices = goal_indices[::-1]
    print(goal_indices)
    for i in goal_indices[:-1]:
        print(extrema_sharpness[i])


    # Filter indices based on subgoal_target if provided
    if subgoal_target is not None and len(goal_indices) > subgoal_target:
        # Always keep the last index (end point)
        if subgoal_target >= 1:
            # Get all indices except the last one, with their sharpness
            candidate_indices = goal_indices[:-1]
            indices_with_sharpness = [
                (idx, extrema_sharpness.get(idx, 0)) for idx in candidate_indices
            ]

            # Sort by sharpness (higher values first)
            sorted_indices = [
                idx for idx, _ in sorted(
                    indices_with_sharpness,
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

            # Take top (subgoal_target-1) indices by sharpness
            top_indices = sorted_indices[:subgoal_target - 1]

            # Add the last index and sort to maintain chronological order
            filtered_indices = top_indices + [goal_indices[-1]]
            goal_indices = sorted(filtered_indices)
        else:
            # If subgoal_target is 0, just keep the last index
            goal_indices = [goal_indices[-1]]
    if fill_embeddings:
        milestone_embeddings = np.concatenate(
            [embeddings[goal_indices[0], ...][None]]
            + [
                np.full((end - start, *embeddings.shape[1:]), embeddings[end, ...])
                for start, end in zip([0] + goal_indices[:-1], goal_indices)
            ],
            )
        if device is not None:
            milestone_embeddings = U.any_to_torch_tensor(
                milestone_embeddings, device=device
            )
    else:
        milestone_embeddings = None

    return milestone_embeddings, DecompMeta(
        milestone_indices=goal_indices, iter_curves=iter_curves
    )


def goal_idx_from_mask(goal_achieved_mask):
    diff = np.diff(goal_achieved_mask)
    goal_indices = np.where(diff != 0)[0] + 1
    traj_length = goal_achieved_mask.shape[0]
    goal_indices[-1] = traj_length - 1  # last
    goal_indices = goal_indices.tolist()
    return goal_indices


def oracle_decomp(
    embeddings: np.ndarray | torch.Tensor | None,
    goal_achieved_mask: np.ndarray,
    random_skip_ratio: float | None = None,
    linearly_random_skip_lower: float | None = None,
    fill_embeddings: bool = True,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    """Note: embeddings here only has the oracle subgoals, not full trajectory"""
    goal_indices = goal_idx_from_mask(goal_achieved_mask)
    if not fill_embeddings:
        return None, DecompMeta(milestone_indices=goal_indices)

    traj_length = goal_achieved_mask.shape[0]
    assert embeddings.shape[0] < traj_length, embeddings.shape
    milestone_embeddings = (
        torch.empty(
            (goal_achieved_mask.shape[0], *embeddings.shape[1:]),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        if not isinstance(embeddings, np.ndarray)
        else np.empty(
            (goal_achieved_mask.shape[0], *embeddings.shape[1:]),
            dtype=embeddings.dtype,
        )
    )

    assert len(goal_indices) == len(embeddings), goal_indices

    last_embedding = embeddings[-1]
    for i, idx in enumerate(goal_achieved_mask):
        if idx >= embeddings.shape[0]:
            # If the index in the mask is greater than the highest index in the embedding,
            # just use the last row of the embedding
            milestone_embeddings[i] = last_embedding
        else:
            skip = False
            if linearly_random_skip_lower is not None:
                skip = linear_random_skip(
                    cur_step=i,
                    next_goal_step=goal_indices[idx],
                    ratio=random_skip_ratio,
                    progress_lower=linearly_random_skip_lower,
                )
            elif (
                random_skip_ratio is not None
                and 0 < random_skip_ratio < random.random()
            ):
                skip = True
            if skip:
                milestone_embeddings[i] = embeddings[min(idx + 1, len(embeddings) - 1)]
            else:
                milestone_embeddings[i] = embeddings[idx]
    return milestone_embeddings, DecompMeta(milestone_indices=goal_indices)


def random_decomp(
    embeddings: np.ndarray | torch.Tensor,
    num_milestones: int | tuple[int, int],
    fill_embeddings: bool = True,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    if not isinstance(num_milestones, int):
        assert len(num_milestones) == 2, num_milestones
        # by randomly sample from lower and higher bound
        num_milestones = random.randint(*num_milestones)
    traj_length = embeddings.shape[0]
    goal_indices = random.sample(range(traj_length), k=num_milestones)
    goal_indices = list(sorted(goal_indices))
    if fill_embeddings:
        milestone_embeddings = (
            torch.empty_like(
                embeddings, dtype=embeddings.dtype, device=embeddings.device
            )
            if not isinstance(embeddings, np.ndarray)
            else np.empty_like(embeddings, dtype=embeddings.dtype)
        )
        for i, goal_idx in enumerate(goal_indices):
            milestone_embeddings[
                (goal_indices[i - 1] + 1) if i != 0 else 0 : goal_idx + 1
            ] = embeddings[goal_idx]
    else:
        milestone_embeddings = None
    return milestone_embeddings, DecompMeta(milestone_indices=goal_indices)


def equally_decomp(
    embeddings: np.ndarray | torch.Tensor,
    num_milestones: int | tuple[int, int],
    fill_embeddings: bool = True,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    if not isinstance(num_milestones, int):
        assert len(num_milestones) == 2, num_milestones
        # by randomly sample from lower and higher bound
        num_milestones = random.randint(*num_milestones)
    traj_length = embeddings.shape[0]
    indices = np.linspace(0, traj_length - 1, num_milestones + 1, dtype=int)
    if fill_embeddings:
        milestone_embeddings = (
            torch.empty_like(
                embeddings, dtype=embeddings.dtype, device=embeddings.device
            )
            if not isinstance(embeddings, np.ndarray)
            else np.empty_like(
                embeddings,
                dtype=embeddings.dtype,
            )
        )
        for i, goal_idx in enumerate(indices[1:], start=1):
            milestone_embeddings[
                (indices[i - 1] + 1) if i != 1 else 0 : goal_idx + 1
            ] = embeddings[goal_idx]
    else:
        milestone_embeddings = None
    return milestone_embeddings, DecompMeta(milestone_indices=indices[1:].tolist())


def near_future_decomp(
    embeddings: np.ndarray | torch.Tensor, advance_steps: int, **kwargs
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    return equally_decomp(
        embeddings, num_milestones=embeddings.shape[0] // advance_steps, **kwargs
    )


def no_decomp(
    embeddings: np.ndarray | torch.Tensor, fill_embeddings: bool = True
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    """Only conditioned on final goal."""
    if not fill_embeddings:
        return None, DecompMeta(milestone_indices=[-1])
    return embeddings[-1, ...].expand_as(embeddings).clone() if not isinstance(
        embeddings, np.ndarray
    ) else np.full(
        embeddings.shape,
        embeddings[-1, ...],
        dtype=embeddings.dtype,
    ), DecompMeta(
        milestone_indices=[-1]
    )


def decomp_trajectories(
    method_name: Literal[
        "embed", "embed_no_robot", "oracle", "random", "equally", "near_future"
    ]
    | None,
    embeddings: np.ndarray | torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    assert embeddings.ndim == 2 or embeddings.ndim == 4, (
        f"input embedding should be either 2 dimensional, "
        f"with (L, feature_dim), or raw rgb with shape (L, H, W, 3), "
        f"but get {embeddings.shape}"
    )
    if method_name is None:
        return no_decomp(embeddings)
    assert method_name in DEFAULT_DECOMP_KWARGS, method_name
    method_kwargs = DEFAULT_DECOMP_KWARGS[method_name]
    method_kwargs.update(kwargs)
    if method_name == "embed":
        return embedding_decomp(embeddings=embeddings, **method_kwargs)
    elif method_name == "embed_no_robot":
        return embed_decomp_no_robot(embeddings=embeddings, **method_kwargs)
    elif method_name == "embed_no_robot_extended":
        return embed_decomp_no_robot_extended(embeddings=embeddings, **method_kwargs)
    elif method_name == "oracle":
        return oracle_decomp(embeddings, **method_kwargs)
    elif method_name == "random":
        return random_decomp(embeddings, **method_kwargs)
    elif method_name == "equally":
        return equally_decomp(embeddings, **method_kwargs)
    elif method_name == "near_future":
        return near_future_decomp(embeddings, **method_kwargs)
    raise NotImplementedError(method_name)


DEFAULT_DECOMP_KWARGS = dict(
    embed=dict(
        normalize_curve=True,
        # min_interval=5,
        min_interval=1,
        smooth_method="kernel",
        gamma=0.04,
        subgoal_target=None,
    ),
    embed_no_robot=dict(
        window_length=8,
        derivative_order=1,
        derivative_threshold=1e-3,
        threshold_subgoal_passing=None,
    ),
    embed_no_robot_extended=dict(
        window_length=3,
        derivative_order=1,
        derivative_threshold=1e-3,
        threshold_subgoal_passing=None,
    ),
    oracle=dict(),
    random=dict(num_milestones=(3, 6)),
    equally=dict(num_milestones=(3, 6)),
    near_future=dict(advance_steps=5),
)
