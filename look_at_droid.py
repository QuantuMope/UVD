import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import imageio
import os


def as_mp4(images, path="output.mp4", fps=15):
    """
    Convert a list of PIL images to an MP4 video file.

    Args:
        images: List of PIL Image objects
        path: Output path for the MP4 file
        fps: Frames per second (default: 15)

    Returns:
        Path to the created MP4 file
    """
    # Convert PIL images to numpy arrays
    numpy_images = [np.array(img) for img in images]

    # Create MP4 video
    imageio.mimsave(path, numpy_images, fps=fps)

    return path


# Load dataset
ds = tfds.load("droid_100", data_dir="/home/asjchoi/Downloads/droid", split="train")

# Collect images from a random episode
images = []
for episode in ds.shuffle(98, seed=0).take(1):
    for i, step in enumerate(episode["steps"]):
        images.append(
            Image.fromarray(
                np.concatenate((
                    step["observation"]["exterior_image_1_left"].numpy(),
                    # step["observation"]["exterior_image_2_left"].numpy(),
                    # step["observation"]["wrist_image_left"].numpy(),
                ), axis=1)
            )
        )

# Generate MP4 file
output_path = as_mp4(images, "droid_visualization.mp4")
print(f"MP4 video created at: {os.path.abspath(output_path)}")

# If you're in a notebook and want to display the video:
# from IPython.display import Video
# Video(output_path)