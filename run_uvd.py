import uvd
import decord
import torch
import numpy as np
import matplotlib.pyplot as plt
import click


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
@click.option("--video_path", type=str, required=True)
@click.option("--subgoal_target", type=int, default=None)
@click.option("--orig_method", is_flag=True, default=False)
def main(video_path, subgoal_target, orig_method):
    frame_original_res = decord.VideoReader(video_path)[:].asnumpy()
    frame_resized_res = decord.VideoReader(video_path, height=224, width=224)[:].asnumpy()
    indices, iter_curves = uvd.get_uvd_subgoals(
        frame_resized_res,
        preprocessor_name='dinov2',
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_indices=True,
        subgoal_target=subgoal_target,
        orig_method=orig_method,
        return_intermediate_curves=True,
    )
    print(indices)
    print(len(iter_curves))
    print(iter_curves[0].shape)
    print(iter_curves[1].shape)

    curve = iter_curves[0]
    plt.figure()
    plt.plot(np.arange(len(curve)), curve)
    for i in indices:
        plt.axvline(i, color='r', linestyle='dashed')

    subgoals = frame_original_res[indices]
    # concatenated_images = np.concatenate(subgoals, axis=1)

    display_subgoals_grid(subgoals)

if __name__ == '__main__':
    main()