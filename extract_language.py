import os
import joblib
import json
import multiprocessing as mp
from functools import partial
from tqdm import tqdm  # Optional for progress tracking


def process_file(fn, data_directory):
    """Process a single file and return the filename and extracted language."""
    try:
        data = joblib.load(os.path.join(data_directory, fn))
        language = data['steps'][0]['language_instruction'].decode('utf-8')
        return fn, language
    except Exception as e:
        print(f"Error processing {fn}: {e}")
        return fn, None


def process_directory(data_directory, json_file_name, num_workers=None):
    """Process all files in a directory in parallel and save results to a JSON file."""
    files = os.listdir(data_directory)

    # Use all available cores if num_workers not specified
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Create a pool of workers
    pool = mp.Pool(processes=num_workers)

    # Prepare the function with fixed data_directory
    process_func = partial(process_file, data_directory=data_directory)

    # Process files in parallel and collect results
    # Using tqdm for progress tracking (optional)
    results = list(tqdm(pool.imap(process_func, files), total=len(files), desc=f"Processing {data_directory}"))

    # Close the pool
    pool.close()
    pool.join()

    # Filter out None results and convert to dictionary
    language_data = {fn: lang for fn, lang in results if lang is not None}

    # Save results to JSON file
    with open(json_file_name, "w") as f:
        json.dump(language_data, f)

    print(f"Processed {len(language_data)} files and saved to {json_file_name}")


if __name__ == "__main__":
    data_directories = [
        "/home/asjchoi/datasets/bridge_unambig/val/",
        "/home/asjchoi/datasets/bridge_unambig/train/"
    ]
    json_file_names = [
        "val_language.json",
        "train_language.json"
    ]

    # Process each directory (sequentially, but files within a directory are processed in parallel)
    for data_directory, json_file_name in zip(data_directories, json_file_names):
        process_directory(data_directory, json_file_name)

    # Alternative: Process directories in parallel too (comment out the loop above and uncomment below)
    # pool = mp.Pool(processes=len(data_directories))
    # args = [(d, j) for d, j in zip(data_directories, json_file_names)]
    # pool.starmap(process_directory, args)
    # pool.close()
    # pool.join()