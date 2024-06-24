import os
import numpy as np
import pandas as pd
import cv2 as cv 

import pickle

from sklearn.ensemble import RandomForestClassifier
 
from feature_extrac import get_features

def get_labels(mask_path):

    mask_dataset = [] 
    for file in sorted(os.listdir(mask_path)): 
        
        mask = cv.imread(mask_path + file)

        label = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        label_values = label.reshape(-1)
        
        mask_dataset.extend(label.reshape(-1))

    mask_dataset = np.asarray(mask_dataset)

    return mask_dataset

if __name__ == '__main__':

    # Training
    train_image_path = "./data/10_dataset/train/images/"
    train_mask_path = "./data/10_dataset/train/masks/"

    Y_train = get_labels(train_mask_path)
    X_train = get_features(train_image_path)

    print("feature extract is done")

    # n is a number of decision trees
    model = RandomForestClassifier(n_estimators = 50, random_state = 42)

    model.fit(X_train, Y_train)

    # Save model
    model_name = "skin_model"
    pickle.dump(model, open(model_name, 'wb'))


    # Feature list
    feature_list = list(X_train.columns)
    feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    print(feature_imp.to_string())




