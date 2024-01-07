
import unittest
import os
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
from feature_extraction import preprocess_image, normalise_features, extract_features


def create_temporary_image():

    temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img = Image.new('RGB', (500, 750), color='white')
    img.save(temp_image.name)
    path = temp_image.name
    temp_image.close()

    return path


class TestPreprocessImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_path = create_temporary_image()

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.image_path)  # Remove the temporary file

    def test_preprocess_image(self):
        preprocessed_img = preprocess_image(self.image_path)
        self.assertEqual(preprocessed_img.shape, (1, 224, 224, 3))


class TestNormaliseFeatures(unittest.TestCase):

    def test_normalise_features(self):
        features = np.array([1, 2, 3])
        normalised = normalise_features(features)
        norm = np.linalg.norm(normalised)
        self.assertAlmostEqual(norm, 1.0)


class TestExtractFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_path = create_temporary_image()

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.image_path)  # Remove the temporary file

    @patch('feature_extraction.preprocess_image')
    @patch('feature_extraction.extract_features_from_model')
    @patch('feature_extraction.normalise_features')
    def test_extract_features(self, mock_normalise, mock_extract_model, mock_preprocess):
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        mock_extract_model.return_value = np.zeros((1, 2048))
        mock_normalise.return_value = np.zeros(2048)

        # Test extracting features from the temporary file
        features, img_id = extract_features(self.image_path, img_id=123)
        self.assertEqual(features.shape, (2048,))
        self.assertEqual(img_id, 123)

        # Test that ValueError is raised for an inaccessible URL
        with self.assertRaises(ValueError):
            extract_features('http://random.com/image.jpg', img_id=123, from_url=True)


if __name__ == '__main__':

    unittest.main()

