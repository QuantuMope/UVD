import uvd
import decord
import torch
import numpy as np
import matplotlib.pyplot as plt
import click


@click.command()
@click.option("--video_path", type=str, required=True)
@click.option("--subgoal_target", type=int, default=None)
@click.option("--orig_method", is_flag=True, default=False)
def main(video_path, subgoal_target, orig_method):
    frame_original_res = decord.VideoReader(video_path)[:].asnumpy()
    frame_resized_res = decord.VideoReader(video_path, height=224, width=224)[:].asnumpy()
    indices = uvd.get_uvd_subgoals(frame_resized_res,
                                   preprocessor_name='dinov2',
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_indices=True,
        subgoal_target=subgoal_target,
        orig_method=orig_method
    )
    print(indices)

    subgoals = frame_original_res[indices]
    concatenated_images = np.concatenate(subgoals, axis=1)
    plt.imshow(concatenated_images)
    plt.show()

if __name__ == '__main__':
    main()