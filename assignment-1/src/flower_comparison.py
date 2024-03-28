import os 
import glob
import pandas as pd
import cv2
import numpy as np 
import argparse

def get_arguments():
    """
    Define the flower image that should be compared to all flower images
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--flower",
                        "-f", 
                        required = False,
                        default = "image_0001.jpg",
                        help="The file name of the flower of choice, the default is image_0001.jpg")
                   
    args = parser.parse_args()
    return args

def image_hist(image):
    """
    The function takes a cv2 image object and returns the normalised histogram
    """
    hist = cv2.calcHist([image], channels = [0,1,2], mask = None, histSize = [255,255,255], ranges = [0,256, 0,256,0,256])
    norm_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    return norm_hist

def pick_flower(image_name):
    """
    The function takes the filename of the image that should be compared to all images
    and returns an outpath with the image name, and the normalised histogram
    """
    filepath_f1 = os.path.join("in", "flowers", image_name)
    image_f1 = cv2.imread(filepath_f1)
    filename_f1 = filepath_f1.split("/")[-1].split(".jpg")[0]
    norm_hist_f1 = image_hist(image_f1)
    outpath_f1 = os.path.join("out", f"{filename_f1}_dist.csv")
    return outpath_f1, norm_hist_f1


def compare_flower(outpath_f1, norm_hist_f1):
    filepath_f = os.path.join("in", "flowers")
    dist_df = pd.DataFrame(columns= ("Filename", "Distance"))

    for file in sorted(os.listdir(filepath_f)):
        filepath_fx = os.path.join(filepath_f, file)
        image_fx = cv2.imread(filepath_fx)
        filename_fx = file.split(".jpg")[0]
        norm_hist_fx = image_hist(image_fx)
        
        dist_fx = round(cv2.compareHist(norm_hist_f1, norm_hist_fx, cv2.HISTCMP_CHISQR),5)
        row_fx = [filename_fx, dist_fx]
        
        if len(dist_df.index)<6:
            dist_df.loc[len(dist_df)] = row_fx
        else:
            max_dist_row = dist_df['Distance'].idxmax()
            max_dist_value = dist_df.loc[max_dist_row, 'Distance']
            if dist_fx<max_dist_value:
                dist_df.loc[max_dist_row] = row_fx

    dist_df.to_csv(outpath_f1) 

def main():
    args = get_arguments()
    outpath_f1, norm_hist_f1 = pick_flower(args.flower)
    compare_flower(outpath_f1, norm_hist_f1)


if __name__ == "__main__":
    main()