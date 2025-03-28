import numpy as np
import joblib
import cv2
import os


def save_video_from_arrays(arrays, output_path, fps=30, codec='mp4v'):
    """
    Save a sequence of numpy arrays as a video file

    Parameters:
    arrays (list): List of numpy arrays (frames)
    output_path (str): Path where the video will be saved
    fps (int): Frames per second
    codec (str): FourCC codec code
    """
    # Get dimensions from the first frame
    height, width = arrays[0].shape[:2]

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame in arrays:
        # OpenCV expects BGR format, convert if your array is RGB
        if frame.ndim == 3 and frame.shape[2] == 3:
            # Convert RGB to BGR if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            pass

        # Ensure the frame is uint8 with values 0-255
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    # directory = "/home/asjchoi/datasets/openx/droid/"
    directory = "/home/asjchoi/datasets/openx/bridge/test/"

    if "droid" in directory:
        image_key = "image"
    elif "bridge" in directory:
        image_key = "image_0"

    for fn in os.listdir(directory):
        data = joblib.load(directory + fn)

        frames = []
        actions = []
        language = data['steps'][0]['language_instruction']

        for time_step in data['steps']:
            img = time_step['observation'][image_key]
            actions.append(time_step['action'])
            frames.append(img)

        # cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        frames = np.array(frames)
        actions = np.array(actions)

        prefix = fn.split(".joblib.gz")[0]
        new_data_name = prefix + ".npz"
        new_video_name = prefix + ".mp4"

        np.savez(directory + new_data_name, frames=frames, actions=actions, language=language)

        save_video_from_arrays(frames, directory + new_video_name, fps=5)




