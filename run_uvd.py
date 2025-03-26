from scipy.signal import argrelextrema

import uvd
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import click
from uvd.decomp.kernel_reg import KernelRegression


def display_subgoals_grid(subgoals, max_cols=3):
    N, H, W, C = subgoals.shape

    # Calculate number of rows needed
    num_rows = (N + max_cols - 1) // max_cols

    # Create an empty canvas for the grid
    # Make it large enough to fit all images
    grid_height = num_rows * H
    grid_width = max_cols * W
    grid = np.zeros((grid_height, grid_width, C), dtype=subgoals.dtype)

    # Fill the grid with subgoal images
    for i in range(N):
        row_idx = i // max_cols
        col_idx = i % max_cols

        y_start = row_idx * H
        x_start = col_idx * W

        grid[y_start:y_start + H, x_start:x_start + W] = subgoals[i]

    # Display the grid
    plt.figure(figsize=(15, 15 * num_rows / max_cols))
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


@click.command()
@click.option("--data_path", type=str, required=True)
@click.option("--subgoal_target", type=int, default=None)
@click.option("--orig_method", is_flag=True, default=False)
def main(data_path, subgoal_target, orig_method):
    # frame_original_res = decord.VideoReader(video_path)[:].asnumpy()
    # frame_resized_res = decord.VideoReader(video_path, height=224, width=224)[:].asnumpy()

    data = np.load(data_path)
    frames = data['frames']
    actions = data['actions']
    language = data['language'].item().decode('utf-8')

    frame_resized_res = np.zeros((frames.shape[0], 224, 224, 3))
    for i in range(frames.shape[0]):
        frame_resized_res[i] = cv2.resize(frames[i], (224, 224))
    frame_resized_res = frame_resized_res.astype(np.float32)

    frame_original_res = frames
    #
    indices, iter_curves = uvd.get_uvd_subgoals(
        frame_resized_res,
        preprocessor_name='dinov2',
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_indices=True,
        subgoal_target=subgoal_target,
        orig_method=orig_method,
        return_intermediate_curves=True,
    )
    print(f"Language: {language}")
    # print(indices)
    # print(len(iter_curves))
    # print(iter_curves[0].shape)
    # print(iter_curves[1].shape)

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    # curve = iter_curves[0]
    x_axis = np.arange(actions.shape[0])

    delta_actions = np.zeros_like(actions)
    delta_actions[1:] = actions[1:] - actions[:-1]

    smooth_delta_actions = np.zeros_like(delta_actions)
    for dim in range(smooth_delta_actions.shape[-1]):
        kr = KernelRegression(gamma=0.04)
        kr.fit(x_axis.reshape(-1, 1), 50 * delta_actions[:, dim])
        smooth_delta_actions[:, dim] = kr.predict(x_axis.reshape(-1, 1))

    abs_sum_delta = np.abs(smooth_delta_actions)[:, :3]
    abs_sum_delta = abs_sum_delta.sum(axis=-1)

    plt.figure()
    # for dim in range(actions.shape[-1]):
    dofs = ['x', 'y', 'z']
    for dim in range(3):  # just x y z
        dof = dofs[dim]
        # plt.plot(x_axis, smooth_delta_actions[:, dim], label=dof)

    plt.plot(x_axis, actions[:, -1], label="gripper", linewidth=5)
    plt.plot(x_axis, abs_sum_delta, label="abs sum delta", linewidth=5)

    # plt.plot(x_axis, normalize(curve), linewidth=5, label="embedding distance")
    # for i in indices:
    #     plt.axvline(i, color='r', linestyle='dashed')
    # plt.legend()
    # plt.plot(x_axis, normalize(curve), linewidth=5, label="embedding distance")

    # indices = argrelextrema(abs_sum_delta, comparator=np.less)[0]
    # indices = argrelextrema(actions[:, -1], comparator=np.less_equal)[0]
    for i in indices:
        plt.axvline(i, color='r', linestyle='dashed')
    plt.legend()


    subgoals = frame_original_res[indices]

    display_subgoals_grid(subgoals)

if __name__ == '__main__':
    main()