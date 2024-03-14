import os 
import sys 
import argparse
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def get_arguments():
    """
    Get arguments used in functions, the default arguments are the ones used in 
    completing assignment 2 for Visual Analytics (CDS, 2024)
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
                        "--outpath",
                        "-o", 
                        required = False,
                        default = "../out/LR_classification_rep.txt",
                        help="The path for the classification report")

    parser.add_argument(
                        "--imageloader",
                        "-i", 
                        required = False,
                        default = cifar10.load_data(),
                        help= "The function used to load in X_train, y_train, X_test and y_test")
    
    args = parser.parse_args()
    return args


def image_loader(load_fun):
    """
    The function returns X_train, y_train, X_test and y_test, as formatted when 
    using the default imageloader 'cifar10.load_data()'. 
    """
    (X_train, y_train), (X_test, y_test) = load_fun
    return X_train, y_train, X_test, y_test


def preprocess(images):
    """
    The function returns the processed version of the input images (greyed, 
    scales and reshaped). The input images must be formatted as a 4D ndarray, 
    [number of images, number of pixels, number of pixels, colour channels (RGB)]
    """
    greyed_shape = (images.shape[0:3])
    greyed_dtype = (images.dtype)
    images_grey = np.empty(shape = greyed_shape, dtype= greyed_dtype)    
    for i in range(0, len(images)):
        image = images[i][:,:,:]
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_grey[i] = image_grey
    images_scaled = images_grey/255.0 
    preprocessed_images = images_scaled.reshape(-1, 1024)

    return preprocessed_images






def main():
    args = get_arguments()
    X_train, y_train, X_test, y_test = image_loader(args.imageloader)
    X_train_pre = preprocess(X_train)
    X_test_pre = preprocess(X_test)
    LR_classifier(X_train_pre, y_train, X_test_pre, y_test, args.outpath)
    

if __name__ == "__main__":
    main()