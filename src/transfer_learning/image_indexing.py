
import numpy as np
from transfer_learning.feature_extraction import extract_features


def find_similar_images(processed_images: dict, dataset_id: str, img_url: str, K: int) -> list:
    """
    Finds and sorts the top K similar images from a local collection compared to
    an online image.

    The function first extracts features from the online image specified by its URL.
    It then calculates the cosine similarity between the features of the online image
    and each image in the local collection.

    The function returns a list of image IDs from the local collection, sorted by
    their similarity score in descending order, representing the most similar images.

    Parameters:
    - processed_images (dict): A dictionary mapping local image IDs to their feature vectors.
    - dataset_id (str): The name of image dataset.
    - img_url (str): The URL of the online image to compare against.
    - K (int): The number of top similar images to return.

    Returns:
    - list: A list of tuples, each containing an image ID from the local collection and its
            corresponding similarity score, sorted in descending order of similarity.
    """

    if not processed_images:
        raise ValueError("The dictionary is empty")

    if dataset_id not in processed_images:
        raise KeyError(f"'{dataset_id}' not found in the dictionary.")

    online_img_features, _ = extract_features(input_arg=img_url, from_url=True)

    similarity_scores = {}

    for img_id, local_img_features_id in processed_images[dataset_id].items():

        # Calculate the cosine similarity
        dot_product = np.dot(online_img_features, local_img_features_id)
        norm_online = np.linalg.norm(online_img_features)
        norm_local = np.linalg.norm(local_img_features_id)
        cosine_similarity_manual = dot_product / (norm_online * norm_local)

        similarity_scores[img_id] = cosine_similarity_manual

    # Sort and select top K similar images
    top_k_images = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:K]

    return top_k_images
