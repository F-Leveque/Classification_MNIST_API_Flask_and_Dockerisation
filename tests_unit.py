import unittest
from skimage.io import imread
import numpy as np
from sklearn.pipeline import Pipeline
from backend import Transformer_deskew
from joblib import dump, load

class Testing(unittest.TestCase):
    def setUp(self) -> None:
        self.p = Transformer_deskew()
        self.i1 = imread("./images_test/images_test_1.jpg")
        self.i2 = imread("./images_test/images_test_2.jpg")
        self.image_deskewed = self.p.transform(self.i1)
        self.image_list_deskewed = self.p.transform(np.array([self.i1, self.i2]))
        self.model = load('model.joblib') 

    def test_transformer_well_declared(self):
        self.assertIsInstance(self.p, Transformer_deskew)

    def test_image_well_imported(self):
        self.assertIsInstance(self.i1, np.ndarray)

    def test_image_shape(self):
        self.assertEqual(self.i1.shape, np.ones((28,28)).shape)

    def test_image_deskewed_shape(self):
        self.assertEqual(self.image_deskewed.shape, np.ones((1,784)).shape)

    def test_image_deskewed_list_shape(self):
        self.assertEqual(self.image_list_deskewed.shape, np.ones((2, 784)).shape)

    def model_not_empty(self):
        self.assertIsNone(self.model)

if __name__ == '__main__':
    unittest.main()