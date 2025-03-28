from scipy.signal import argrelextrema
import uvd
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import click
from gripper_filter import detect_gripper_transitions
from uvd.decomp.kernel_reg import KernelRegression
from run_uvd import display_subgoals_grid
from uvd.decomp import *
from uvd.models import *


@click.command()
@click.option("--data_path", type=str, required=True)
@click.option("--subgoal_target", type=int, default=None)
def main(data_path, subgoal_target):
    data = np.load(data_path)
    frames = data['frames']
    actions = data['actions']
    language = data['language'].item().decode('utf-8')
    print(f"Language: {language}")

    x = np.arange(actions.shape[0])

    delta_actions = np.zeros_like(actions)
    delta_actions[1:] = actions[1:] - actions[:-1]

    smooth_delta_actions = np.zeros_like(delta_actions)
    for dim in range(smooth_delta_actions.shape[-1]):
        kr = KernelRegression(gamma=0.04)
        kr.fit(x.reshape(-1, 1), 50 * delta_actions[:, dim])
        smooth_delta_actions[:, dim] = kr.predict(x.reshape(-1, 1))

    abs_sum_delta = np.abs(smooth_delta_actions)[:, :3]
    abs_sum_delta = abs_sum_delta.sum(axis=-1)

    # Calculate directional changes using dot products
    directional_changes = np.zeros(len(delta_actions))

    # We need at least 2 delta_actions to calculate a dot product
    for i in range(2, len(delta_actions)):
        v1 = smooth_delta_actions[i - 1, :3]  # Previous xyz delta
        v2 = smooth_delta_actions[i, :3]  # Current xyz delta

        # Normalize vectors to focus on direction, not magnitude
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        # Avoid division by zero
        if v1_norm > 1e-6 and v2_norm > 1e-6:
            v1_unit = v1 / v1_norm
            v2_unit = v2 / v2_norm

            # Cosine similarity between consecutive delta actions
            cos_sim = np.dot(v1_unit, v2_unit)

            # Convert to angle (in radians)
            angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))

            # Store the angular change (higher value means more directional change)
            directional_changes[i] = angle

    xyz_position = actions[:, :3]
    xyz_position = np.zeros((xyz_position.shape[0], 4))
    xyz_position[:, :3] = actions[:, :3]
    gripper_actions = actions[:, -1]
    kr = KernelRegression(gamma=0.04)
    kr.fit(x.reshape(-1, 1), gripper_actions)
    smooth_gripper_actions = kr.predict(x.reshape(-1, 1))
    gripper_indices, gripper_signal = detect_gripper_transitions(gripper_actions, window_size=5, invert_state=True)

    # xyz_position[:, 3] = actions[:, -1] * 0.25
    # xyz_position[:, 3] = actions[:, -1] * 1.0
    xyz_position[:, 3] = gripper_signal

    kr = KernelRegression(gamma=0.08)
    kr.fit(x.reshape(-1, 1), gripper_signal)
    smooth_gripper_signal = kr.predict(x.reshape(-1, 1))
    xyz_position[:, 3] = smooth_gripper_signal * 3

    frame_resized_res = np.zeros((frames.shape[0], 224, 224, 3))
    for i in range(frames.shape[0]):
        frame_resized_res[i] = cv2.resize(frames[i], (224, 224))
    frame_resized_res = frame_resized_res.astype(np.float32)

    frame_original_res = frames
    #
    # indices, iter_curves = uvd.get_uvd_subgoals(
    #     frame_resized_res,
    #     preprocessor_name='dinov2',
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     return_indices=True,
    #     subgoal_target=subgoal_target,
    #     orig_method=True,
    #     return_intermediate_curves=True,
    # )

    preprocessor = get_preprocessor("dinov2", device="cuda")
    rep = preprocessor.process(frame_resized_res, return_numpy=True)

    curr_index = len(xyz_position)
    indices = []
    embedding = []
    embedding_diff = []
    xyz_dist = np.linalg.norm(xyz_position - xyz_position[-1], axis=-1)
    iter = -1
    while True:
        segment = xyz_position[:curr_index]
        seg_xyz_dist = np.linalg.norm(segment - segment[-1], axis=-1)
        #
        kr = KernelRegression(gamma=0.04)
        kr.fit(x[:curr_index].reshape(-1, 1), seg_xyz_dist)
        seg_xyz_dist = kr.predict(x[:curr_index].reshape(-1, 1))
        # seg_xyz_dist = directional_changes[:curr_index]

        # seg_xyz_dist = abs_sum_delta[:curr_index]
        # seg_xyz_dist = seg_xyz_dist - seg_xyz_dist[-1]


        extrema = argrelextrema(seg_xyz_dist, comparator=np.greater, order=5)[0]
        # extrema2 = argrelextrema(seg_xyz_dist, comparator=np.less)[0]
        # extrema = np.union1d(extrema, extrema2)

        if len(extrema) == 0:
            break

        if len(embedding) != 0:
            embedding_dist = np.linalg.norm(rep[extrema[-1]] - embedding[-1][1])
            embedding_diff.append((iter, embedding_dist))

        embedding.append((iter, rep[extrema[-1]]))
        indices.append(extrema[-1])
        curr_index = extrema[-1]

        iter += 1

        # plt.plot(np.arange(len(seg_xyz_dist)), seg_xyz_dist)
        # plt.show()

    if subgoal_target is not None:
        embedding_sorted = sorted(embedding_diff, key=lambda x: x[1], reverse=True)
        print(embedding_sorted)

    # exit(0)


    plt.plot(x, 3*xyz_dist, label="xyz_dist", linewidth=5)
    plt.plot(x, gripper_actions, label="gripper action", linewidth=5)
    plt.plot(x, gripper_signal, label="gripper signal", linewidth=5)
    plt.plot(x, smooth_gripper_signal, label="smooth gripper signal", linewidth=5)
    plt.plot(x, smooth_gripper_actions, label="smooth gripper action", linewidth=5)
    # plt.plot(x, directional_changes, label="direction change", linewidth=5)
    # plt.plot(x, abs_sum_delta, label="abs sum delta", linewidth=5)
    # plt.plot(x, iter_curves[0], label="embedding", linewidth=5)


    # indices = list(indices[::-1])
    # indices.append(len(xyz_position) - 1)
    # indices = np.array(indices)

    gripper_indices += 2
    indices = list(gripper_indices)
    indices.append(len(xyz_position) - 1)
    indices = np.array(indices)

    for i in indices:
        plt.axvline(x=i, linestyle="--")
    plt.tight_layout()
    plt.legend(fontsize=26)
    display_subgoals_grid(frames[indices])
    plt.show()

    # xyz_position = xyz_position[::-1]
    # rep = rep[::-1]
    # curr_index = len(xyz_position)
    # indices = []
    # embedding = []
    # embedding_diff = []
    # xyz_dist = np.linalg.norm(xyz_position - xyz_position[-1], axis=-1)
    # iter = -1
    # while True:
    #     segment = xyz_position[:curr_index]
    #     seg_xyz_dist = np.linalg.norm(segment - segment[-1], axis=-1)
    #     #
    #     kr = KernelRegression(gamma=0.04)
    #     kr.fit(x[:curr_index].reshape(-1, 1), seg_xyz_dist)
    #     seg_xyz_dist = kr.predict(x[:curr_index].reshape(-1, 1))
    #     # seg_xyz_dist = directional_changes[:curr_index]
    #
    #     # seg_xyz_dist = abs_sum_delta[:curr_index]
    #     # seg_xyz_dist = seg_xyz_dist - seg_xyz_dist[-1]
    #
    #
    #     extrema = argrelextrema(seg_xyz_dist, comparator=np.greater)[0]
    #     # extrema2 = argrelextrema(seg_xyz_dist, comparator=np.less)[0]
    #     # extrema = np.union1d(extrema, extrema2)
    #
    #     if len(extrema) == 0:
    #         break
    #
    #     if len(embedding) != 0:
    #         embedding_dist = np.linalg.norm(rep[extrema[-1]] - embedding[-1][1])
    #         embedding_diff.append((iter, embedding_dist))
    #
    #     embedding.append((iter, rep[extrema[-1]]))
    #     indices.append(extrema[-1])
    #     curr_index = extrema[-1]
    #
    #     iter += 1
    #
    #     plt.plot(np.arange(len(seg_xyz_dist)), seg_xyz_dist)
    #     plt.show()
    #
    # if subgoal_target is not None:
    #     embedding_sorted = sorted(embedding_diff, key=lambda x: x[1], reverse=True)
    #     print(embedding_sorted)
    #
    # # exit(0)
    #
    #
    # plt.plot(x, 3*xyz_dist, label="xyz_dist", linewidth=5)
    # plt.plot(x, gripper_actions, label="gripper action", linewidth=5)
    # # plt.plot(x, directional_changes, label="direction change", linewidth=5)
    # # plt.plot(x, abs_sum_delta, label="abs sum delta", linewidth=5)
    # # plt.plot(x, iter_curves[0], label="embedding", linewidth=5)
    #
    # indices = list(indices[::-1])
    # indices.append(len(xyz_position) - 1)
    # indices = np.array(indices)
    #
    # for i in indices:
    #     plt.axvline(x=i, linestyle="--")
    # plt.tight_layout()
    # plt.legend(fontsize=26)
    # display_subgoals_grid(frames[indices])
    # plt.show()


if __name__ == "__main__":
    main()
