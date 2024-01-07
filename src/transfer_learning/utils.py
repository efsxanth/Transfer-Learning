
import numpy as np


def generate_random_ids(num_images: int) -> np.array:
    """
    Generates a shuffled array of unique IDs for a given number of images.

    The random seed is fixed to 42 to ensure reproducibility of the results
    across different runs.

    Parameters:
    - num_images (int): The number of images for which to generate unique IDs.

    Returns:
    - np.array: An array of unique, randomly shuffled integer IDs ranging from 1 to num_images.
    """

    img_unique_ids = np.arange(1, num_images + 1)
    np.random.seed(42)
    np.random.shuffle(img_unique_ids)

    return img_unique_ids
