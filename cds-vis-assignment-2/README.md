# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
Visual Analytics, Cultural Data Science, 2024 
Date: *15/03/2024*
By: Laura Givskov Rahbek 

## Requirements and description for assignment 2
To run the code, response to Assignment 2 prompt, the following packages should be installed: ```tensorflow```, ```opencv-python```, ```matplotlib```, ```scikit-learn```. 

The folder ```cds-vis-assignment-2``` contains the following: 
- The ```out``` folder with classification reports for the logistic regression classifier and the MLP classifier, as well as a plot of the loss curve for the MLP classifier. 
- The ```src``` folder with two .py scripts 
    - LR_classifier.py: Preprocess, fits and tests a logistic regression classifier on images, the default images are the ones from ```Cifar10``` dataset. The script returns a classification report to a given outpath, the default is the ```out``` folder.
    - MLP_classifier.py: Preprocess, fits and tests a MLP classifier on images, the default images are the ones from ```Cifar10``` dataset. The script returns a classification report and a plot of the loss curve to a given outpath, the default is the ```out``` folder.


## Assignment prompt
For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset. You can read more about this dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html)

You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, normalize, reshape)
- Train a classifier on the data
    - A logistic regression classifier *and* a neural network classifier
- Save a classification report
- Save a plot of the loss curve during training

You should write **two scripts** for this assignment one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn``` to evaluate model performance.


### Purpose

- To ensure that you can use ```scikit-learn``` to build simple benchmark classifiers on image classification data
- To demonstrate that you can build reproducible pipelines for machine learning projects
- To make sure that you can structure repos appropriately

### Starter code

The data already has a train/test split and can be loaded in the following way:

```python
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

Once you have loaded the data, made it greyscale, and scaled the values then you will need to *reshape* the array to be the correct dimensions - essentially flattening the 2d array like we saw with greyscale histograms. 

You can do that in the following way using ```numpy```:

```python
X_train_scaled.reshape(-1, 1024)
X_test_scaled.reshape(-1, 1024)
```


### Tips

- The Cifar10 dataset you download does not have explict label names but instead has numbers from 0-9. You'll need to make a list of labels based on the object names - you can find these on the website.
- You should structure your project by having scripts saved in a folder called ```src```, and have a folder called ```out``` where you save the classification reports.


