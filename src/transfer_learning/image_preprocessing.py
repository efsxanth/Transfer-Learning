
import os
import numpy as np
import concurrent
from concurrent.futures import ProcessPoolExecutor
from transfer_learning.feature_extraction import extract_features     
from transfer_learning.utils import generate_random_ids         
from transfer_learning.default import FEATURES_FOLDER_NAME, NUM_PROCESSES


def preprocess_images(main_directory_img_database: str, num_processes: int = NUM_PROCESSES) -> dict:
    """
    Preprocesses a collection of images located in a given directory (image database).
    The directory should contain sub-folders, each representing a different dataset ID.
    
    Directory Structure:
    - main_directory_img_database (e.g., 'path/to/image_database'):
        - Subfolder for each dataset (e.g., 'dataset_1'):
            - Image files within each subfolder (e.g., '1.png', '2.png', ...)
    
    path/to/image_database/
    ├── dataset_1/
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── another_dataset/
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── ...

    The function checks for pre-extracted features and processes images accordingly. The
    file names of features correspond to unique IDs assigned when initially images were
    analysed.

    The function first checks for the presence of pre-extracted features. If these features
    are available, their filenames, which correspond to the unique IDs assigned to the images
    when they were initially analysed, are used to map the images to their features.

    If features are not pre-extracted, it extracts them using multiprocessing. For each
    image, features are extracted, and the image is assigned a unique ID.

    The function returns a dictionary where keys are dataset IDs and values are dictionaries
    mapping image IDs to their features.

    Parameters:
    - main_directory_img_database (str): The path to the main directory containing the image database.
    - num_processes (int, optional): The number of processes to use for multiprocessing. Defaults to
                                     the number defined in NUM_PROCESSES.

    Returns:
    - dict: A dictionary where each key is a dataset ID and each value is another dictionary mapping
            image IDs to their extracted features.
    """

    if not main_directory_img_database:
        raise ValueError("The directory to the image database must be provided.")

    sub_folders = os.listdir(main_directory_img_database)

    if not sub_folders:
        raise ValueError("The image database is empty.")

    processed_images_dict = {}

    for dataset_id in sub_folders:

        directory_dataset_id = os.path.join(main_directory_img_database, dataset_id)

        # Check for pre-extracted features
        directory_existing_features = os.path.join(directory_dataset_id, FEATURES_FOLDER_NAME)

        processed_images_dict[dataset_id] = {}

        if os.path.exists(directory_existing_features):

            existing_features_per_id = os.listdir(directory_existing_features)

            if not existing_features_per_id:
                raise FileNotFoundError(f'No pre-extracted feature files found in the '
                                        f'specified directory: {directory_existing_features}')

            for feature_file_id in existing_features_per_id:
                img_id = int(feature_file_id.split('.')[0])
                features = np.load(os.path.join(directory_existing_features, feature_file_id))
                processed_images_dict[dataset_id][img_id] = features

        else:

            img_fnames = os.listdir(directory_dataset_id)  # file names of images

            if not img_fnames:
                raise ValueError(f"The {dataset_id} subfolder of the image database is empty.")

            num_images = len(img_fnames)
            img_unique_ids = generate_random_ids(num_images=num_images)

            # Multiprocessing local images
            with ProcessPoolExecutor(max_workers=num_processes) as executor:

                # Create a list of tasks
                tasks = [(os.path.join(directory_dataset_id, img_fname), img_unique_ids[i])
                         for i, img_fname in enumerate(img_fnames)]

                # Process images in parallel (executor.submit(function, *args))
                futures = [executor.submit(extract_features, input_arg=task[0], img_id=task[1]) for task in tasks]

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    img_features, img_id = future.result()
                    print(f'Preprocessing image: {i + 1}/{num_images} - id: {img_id}')
                    processed_images_dict[dataset_id][img_id] = img_features

    return processed_images_dict
