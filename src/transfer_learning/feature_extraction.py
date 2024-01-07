
import os
import requests
from io import BytesIO
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from typing import Tuple, Union
from transfer_learning.default import SELECT_WEIGHTS, IMG_TARGET_SIZE, FEATURES_FOLDER_NAME


def preprocess_image(input_arg: Union[str, BytesIO]) -> np.array:
    """
    Preprocess the image for the model.

    Parameters:
    - input_arg (Union[str, BytesIO]): The file path of the image
      or a BytesIO object containing the image data.

    Returns:
    - np.ndarray: The preprocessed image as a numpy array ready to be fed into a model.
    """

    img = image.load_img(input_arg, target_size=IMG_TARGET_SIZE)  # Resize the image
    img_array = image.img_to_array(img)  # Convert it to an array
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_preprocess_input = preprocess_input(img_array_expanded_dims)  # data_format=None (default)

    return img_preprocess_input


def extract_features_from_model(preprocessed_img: np.array) -> np.array:
    """
    Extracts features from an image using the ResNet50 pre-trained model.

    This function utilizes transfer learning to leverage the ResNet50 pre-trained
    model for feature extraction.

    Parameters:
    - preprocessed_img (np.ndarray): The preprocessed image input as a numpy array.

    Returns:
    - np.ndarray: A numpy array containing the extracted features from the specified
                  pre-trained model layer.

    References:
    - He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition.
      Retrieved from https://arxiv.org/pdf/1512.03385.pdf

    - Gkelios, S., et al. (2021). Deep Convolutional Features for Image Retrieval.
      Expert Systems With Applications, 177. Retrieved from
      https://www.sciencedirect.com/science/article/pii/S095741742100381X

    - https://keras.io/api/applications/#usage-examples-for-image-classification-models
    """

    base_model = ResNet50(weights=SELECT_WEIGHTS)

    # Adapt model to output from the Average Pooling Layer, just before the fully connected layers.
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    features = model.predict(preprocessed_img, verbose=0)

    return features


def normalise_features(features: np.array) -> np.array:
    """
    Normalises a feature vector using the L2 norm.

    This function first flattens the input feature array and then applies
    L2 normalisation. L2 normalisation scales the feature vector so that
    the sum of the squares of its elements equals 1, which is a common
    practice in feature vector normalisation to ensure that the magnitude
    of the vector does not influence similarity calculations.

    Parameters:
    - features (np.ndarray): The input feature vector.

    Returns:
    - np.ndarray: The L2-normalized feature vector.
    """

    features_flat = features.flatten()

    # Normalise the features with L2 norm
    features_norm = features_flat / np.linalg.norm(features_flat, ord=2)

    return features_norm


def extract_features(input_arg: Union[str, BytesIO],
                     img_id: Union[int, None] = None,
                     from_url: bool = False) -> Tuple[np.ndarray, Union[int, None]]:
    """
    Extracts and normalises features from an image, either from a URL or a local file.

    The function handles different input types: if 'from_url' is True, it retrieves
    the image from the specified URL, otherwise, it expects a local file path.

    After retrieving or verifying the image, it preprocesses the image, extracts
    features using a pre-trained model (transfer learning), and then normalises
    these features.

    If the image is a local file, the extracted features are saved in a specified
    directory. The function returns the normalised features and the image ID.

    Parameters:
    - input_arg (str/BytesIO): The path to the local image file or the URL of the image.
    - img_id (int, optional): The ID of the image, used for saving the features. Defaults to None.
    - from_url (bool, optional): Flag to indicate if the image is to be loaded from a URL. Defaults to False.

    Returns:
    - tuple: A tuple containing the normalised features and the image ID.
    """

    if from_url:
        response = requests.get(input_arg)
        if response.status_code != 200:
            raise ValueError(f"Failed to retrieve image from URL: {input_arg}")
        input_arg = BytesIO(response.content)

    # Handling local files
    elif not os.path.exists(input_arg):
        raise FileNotFoundError(f"No image file found at specified path: {input_arg}")

    preprocessed_img = preprocess_image(input_arg=input_arg)
    features = extract_features_from_model(preprocessed_img=preprocessed_img)
    features_norm = normalise_features(features=features)

    # Save extracted features
    if not from_url:
        dir_features = os.path.join(os.path.dirname(input_arg), FEATURES_FOLDER_NAME)
        os.makedirs(dir_features, exist_ok=True)
        save_path = os.path.join(dir_features, f'{img_id}.npy')
        np.save(save_path, features_norm)

    return features_norm, img_id
