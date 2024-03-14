from tensorflow.keras.datasets import cifar10
import os 
import sys 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import argparse

def get_arguments():
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
    (X_train, y_train), (X_test, y_test) = load_fun
    return X_train, y_train, X_test, y_test


def preprocess(images):
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

def LR_classifier(X_prep_train, y_train, X_prep_test, y_test, outpath):

    fitted_classifier = LogisticRegression(random_state=42).fit(X_prep_train, y_train)
    y_pred = fitted_classifier.predict(X_prep_test)

    metrics_rep = metrics.classification_report(y_test, y_pred)
    filepath_metrics = open(outpath, 'w')
    filepath_metrics.write(metrics_rep)
    filepath_metrics.close()


def main():
    args = get_arguments()
    X_train, y_train, X_test, y_test = image_loader(args.imageloader)
    X_train_pre = preprocess(X_train)
    X_test_pre = preprocess(X_test)
    LR_classifier(X_train_pre, y_train, X_test_pre, y_test, args.outpath)
    

if __name__ == "__main__":
    main()