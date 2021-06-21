#Imports
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import interpolation
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.svm import SVC
import numpy as np
import gzip

def set_images(file):
    """Function to obtain the images contained in the .gz file.

    Args:
        file (file .gz): The first parameter.

    Returns:
        images: np.array, shape (image_count, row_count, column_count) 

    source:
        https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

    """
    with gzip.open('../data/' + file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images

def set_labels(file):
    """Function to obtain the labels contained in the .gz file.

    Args:
        file (file .gz): The first parameter.

    Returns:
        labels: np.array

    source:
        https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

    """
    with gzip.open('../data/' + file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

class Transformer_deskew(BaseEstimator, TransformerMixin):
    """Transformer to preprocess the images before classifier

    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        """Function to deskew the images.

        Args:
            X (np.array): The first parameter.

        Returns:
            np.array: image deskewed
            or list of np.array: list of images deskewed

        """
        def moments(image):
            """Find the center of mass of the image and find the covariance matrix of the image pixel intensities

            Args:
                image (np.array): The first parameter.

            Returns:
                tuple: image deskewed
                or list of tuples: list of images deskewed

            source:
                https://fsix.github.io/mnist/Deskewing.html

            """

            c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
            totalImage = np.sum(image) #sum of pixels
            m0 = np.sum(c0*image)/totalImage #mu_x
            m1 = np.sum(c1*image)/totalImage #mu_y
            m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
            m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
            m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
            mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
            covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
            return mu_vector, covariance_matrix

        def deskew(image):
            """shifts the lines so as to make it vertical, affine transformation.

            Args:
                image (np.array): The first parameter.

            Returns:
                list of tuples: list of images deskewed
                or
                tuple: image deskewed

            source:
                https://fsix.github.io/mnist/Deskewing.html

            """
            c,v = moments(image)
            alpha = v[0,1]/v[0,0]
            affine = np.array([[1,0],[alpha,1]])
            ocenter = np.array(image.shape)/2.0
            offset = c-np.dot(affine,ocenter)
            return interpolation.affine_transform(image,affine,offset=offset)

        # if the shape of the argument is not 2 (meaning a single image) we map the deskew function over the list of image
        if len(X.shape) != 2:
            return np.array(list(map(deskew, X.copy()))).reshape(len(X.copy()),-1)
        else:
            return deskew(X.copy()).reshape(1,-1)

def train_model():
    """Function to train the pipeline model : Transformer + SVM

    Args:
        no Args.

    Returns:
        model.joblib: file of the model

    """
    images_train = set_images('./data/train-images-idx3-ubyte.gz')
    labels_train = set_labels('./data/train-labels-idx1-ubyte.gz')
    tf = Transformer_deskew()
    model = Pipeline([('deskew', Transformer_deskew()), ('SVC', SVC(kernel="poly", degree = 4))])
    model.fit(images_train, labels_train)
    filename = 'model.joblib'
    dump(model, filename)   
    
def predict(x):
    """Function to predict the entries

    Args:
        x (np.array): The first parameter, can be a single image or a np.array of images

    Returns:
        model.predict(x): prediction of the entries

    """
    model = load('model.joblib')    
    return model.predict(x)

if __name__ == "__main__":
    train_model()