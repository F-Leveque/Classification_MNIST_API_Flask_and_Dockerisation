# Overview
The objectif is to develop and package, in a running docker container, a classification application for MNIST data.

# Architecture

All the architecture is encapsulated in a [docker](https://www.docker.com/) container. It allows to any user to install the application, upload an image from test_dataset MNIST and obtain the prediction of the number written on (prediction returned in a json file).

A simple web inteface has been developped (Flask framework) allowing the user to select the files used for the prediction.

# Workflow

At the root of the project, we find the file ```backend.py``` containing the classifier code, the pipeline preprocessing and the training method. 
The Flask application is coded in the file ```app.py```. You find also the Dockerfile (with the requirements) for the creation of the Docker image.

- The Docker container launches the Flask application, accessible with the following command in your web browser :
```
    http://localhost:5000/
```
- The File Selection Form is available and will send to the server the images selected.
- The Classifier (under the ```backend.py``` file) is then called and realize the prediction.
- The server returns the prediction in a json file.

# Requirements

* [Docker Engine](https://docs.docker.com/install/) (version 20.10.6)
* 2.5 GB de RAM

# Setup
In order to launch the creation of the Docker image, follow the instructions below :

- Install [docker](https://www.docker.com/) (version 20.10.6)

- Open a shell in the directory where the file Dockerfile is savedand run the command :
```
    $ docker build -t flask-tutorial:latest .
```
Wait until the process succeded in the creation of the Docker image.

- Run the command to start the container :
```
    $ docker run -d -p 5000:5000 flask-tutorial
```

- Open your browser and use the following adress :
```
    http://localhost:5000/
```

- Use the Form to select your files and obtain your prediction !


# Exemple
### Data used
Classification model (accuracy on test dasaset 98.6%): SVM deg 4 polynomial + deskewing preprocessing
Data available on the website http://yann.lecun.com/exdb/mnist/
Train data : train-images-idx3-ubyte.gz + train-labels-idx1-ubyte.gz
Test data : t10k-images-idx3-ubyte.gz + t10k-labels-idx1-ubyte.gz

- 10 images from the test dataset are available in the folder images_test.

# References
- https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
- https://stackoverflow.com/questions/43577665/deskew-mnist-images
