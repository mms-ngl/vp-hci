import os
import numpy as np
import pandas as pd
import cv2 as cv

import pickle

from feature_extrac import feature_extraction
from sklearn import metrics

filename = "skin_model"

def get_labels(mask_path):

    mask_dataset = [] 
    for file in sorted(os.listdir(mask_path)): 

        mask = cv.imread(mask_path + file)

        label = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        
        mask_dataset.extend(label.reshape(-1))

    mask_dataset = np.asarray(mask_dataset)

    return mask_dataset
    

if __name__ == '__main__':

    model = pickle.load(open(filename, 'rb'))

    # Testing
    test_image_path = "./data/dataset/test/images/"
    test_mask_path = "./data/dataset/test/masks/" 

    segmented_path = "./data/segmented_data/"


    X_test = pd.DataFrame() 
    for file in sorted(os.listdir(test_image_path)):  

        test_image = cv.imread(test_image_path + file)
        
        X = feature_extraction(test_image)

        X_test = pd.concat([X_test, X], axis=0)


    Y_test = get_labels(test_mask_path)
        
    # Accuracy check
    predictions = model.predict(X_test)

    # Check accuracy on test dataset. 
    print("Accuracy = ", metrics.accuracy_score(Y_test, predictions))

    
    