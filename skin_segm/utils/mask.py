import os
import sys
import time
from glob import glob
import random, json, math
import itertools

import shutil

import numpy as np
import cv2

def get_mask(annot_path, mask_path):

    for json_file in glob(annot_path+'/*.json'):

        with open(json_file) as f:
            image_annots = json.load(f)

        name = os.path.splitext(os.path.basename(json_file))[0].split(".")[0]

        image_path = os.path.join(annot_path, name+".jpeg")

        height, width = cv2.imread(image_path).shape[:2]

        annots = image_annots["shapes"]

        mask = np.zeros((height, width, 3), np.uint8)

        for annot in annots:
            points = annot["points"]

            points = np.array(points)

            cv2.fillPoly(mask, pts=[np.array(points).astype(np.int32)], color=(255, 255, 255))

            cv2.imwrite(os.path.join(mask_path, name + ".png"), mask) 


if __name__ == '__main__':

    # For hives
    annot_path = "../data/labelled_images/hives/images/"
    mask_path = "../data/labelled_images/hives/masks/"
    get_mask(annot_path, mask_path)

    # For rashes
    annot_path = "../data/labelled_images/rashes/images"
    mask_path = "../data/labelled_images/rashes/masks"
    get_mask(annot_path, mask_path)
    
