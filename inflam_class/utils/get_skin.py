import os
import cv2
import random
import shutil

if __name__ == '__main__':

    # inflam_detect
    # path = "../data/dataset/inflam_detect/train/inflamed/"
    # path = "../data/dataset/inflam_detect/test/inflamed/"
    # path = "../data/dataset/inflam_detect/val/inflamed/"

    # path = "../data/dataset/inflam_detect/train/normal/"
    # path = "../data/dataset/inflam_detect/test/normal/"
    # path = "../data/dataset/inflam_detect/val/normal/"

    # inflam_classify
    # path = "../data/dataset/inflam_classify/train/dermatitis/"
    # path = "../data/dataset/inflam_classify/test/dermatitis/"
    # path = "../data/dataset/inflam_classify/val/dermatitis/"

    # path = "../data/dataset/inflam_classify/train/hives/"
    # path = "../data/dataset/inflam_classify/test/hives/"
    # path = "../data/dataset/inflam_classify/val/hives/"

    # path = "../data/dataset/inflam_classify/train/rashes/"
    # path = "../data/dataset/inflam_classify/test/rashes/"
    path = "../data/dataset/inflam_classify/val/rashes/"


    save = os.path.join(path, "save")

    counter = 0

    for mask in os.listdir(path):  

        if mask.endswith(".png"):
            counter += 1

            name = os.path.splitext(os.path.basename(mask))[0].split(".")[0]

            mask_path = os.path.join(path, mask)

            mask_img = cv2.imread(mask_path)

            image = None

            if os.path.exists(os.path.join(path, name + ".jpeg")):

                image = cv2.imread(os.path.join(path, name + ".jpeg"))

            elif os.path.exists(os.path.join(path, name + ".jpg")):

                image = cv2.imread(os.path.join(path, name + ".jpg"))
            
            mask_img[mask_img == 255] = 1
            target = image * mask_img

            cv2.imwrite(os.path.join(save, str(counter) + ".png"), target)




                   
    
    