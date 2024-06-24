import os
import cv2 as cv 
import numpy as np
import pandas as pd
import skimage

from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd

from skimage.filters import sobel
from skimage.feature import hog
from skimage.feature import local_binary_pattern

from skimage.filters.rank import entropy
from skimage.morphology import disk

import fast_glcm

def color_features(image):
    df = pd.DataFrame()

    #Color features extractions
    rbg_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rbg_r, rbg_g, rbg_b = cv.split(rbg_image)
    df['rbg_r'] = rbg_r.reshape(-1) 
    df['rbg_g'] = rbg_g.reshape(-1) 
    df['rbg_b'] = rbg_b.reshape(-1)   

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv_h, hsv_s, hsv_v = cv.split(hsv_image)
    df['hsv_h'] = hsv_h.reshape(-1)   
    df['hsv_s'] = hsv_s.reshape(-1)   
    df['hsv_v'] = hsv_v.reshape(-1)  

    luv_image = cv.cvtColor(image, cv.COLOR_BGR2Luv)
    luv_l, luv_u, luv_v = cv.split(luv_image)
    df['luv_l'] = luv_l.reshape(-1)   
    df['luv_u'] = luv_u.reshape(-1)   
    df['luv_v'] = luv_v.reshape(-1)  

    xyz_image = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
    xyz_x, xyz_y, xyz_z = cv.split(xyz_image)
    df['xyz_x'] = xyz_x.reshape(-1)   
    df['xyz_y'] = xyz_y.reshape(-1)   
    df['xyz_z'] = xyz_z.reshape(-1)  

    ycrcb_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    ycrcb_y, ycrcb_cr, ycrcb_cb = cv.split(ycrcb_image)
    df['ycrcb_y'] = ycrcb_y.reshape(-1)   
    df['ycrcb_cr'] = ycrcb_cr.reshape(-1)   
    df['ycrcb_cb'] = ycrcb_cb.reshape(-1) 

    lab_image = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    lab_l, lab_a, lab_b = cv.split(lab_image)
    df['lab_l'] = lab_l.reshape(-1)   
    df['lab_a'] = lab_a.reshape(-1)   
    df['lab_b'] = lab_b.reshape(-1) 

    hls_image = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    hls_h, hls_l, hls_s = cv.split(hls_image)
    df['hls_h'] = hls_h.reshape(-1)   
    df['hls_l'] = hls_l.reshape(-1)   
    df['hls_s'] = hls_s.reshape(-1) 

    yuv_image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    yuv_y, yuv_u, yuv_v = cv.split(yuv_image)
    df['yuv_y'] = yuv_y.reshape(-1)   
    df['yuv_u'] = yuv_u.reshape(-1)   
    df['yuv_v'] = yuv_v.reshape(-1) 

    return df


def gabor_features(gray_image):

    df = pd.DataFrame()

    #Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                    ksize=9
                    kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv.filter2D(gray_image, cv.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label

    return df


def glcm_features(gray_image):
    df = pd.DataFrame()

    #GLCM
    mean = fast_glcm.fast_glcm_mean(gray_image)
    mean = mean.reshape(-1)
    df['glcm_mean'] = mean  

    std = fast_glcm.fast_glcm_std(gray_image)
    std = std.reshape(-1)
    df['glcm_std'] = std  

    cont = fast_glcm.fast_glcm_contrast(gray_image)
    cont = cont.reshape(-1)
    df['glcm_cont'] = cont  

    diss = fast_glcm.fast_glcm_dissimilarity(gray_image)
    diss = diss.reshape(-1)
    df['glcm_diss'] = diss  

    homo = fast_glcm.fast_glcm_homogeneity(gray_image)
    homo = homo.reshape(-1)
    df['glcm_homo'] = homo  

    asm, ene = fast_glcm.fast_glcm_ASM(gray_image)
    asm = asm.reshape(-1)
    df['glcm_asm'] = asm  
    ene = ene.reshape(-1)
    df['glcm_ene'] = ene  

    ma = fast_glcm.fast_glcm_max(gray_image)
    ma = ma.reshape(-1)
    df['glcm_ma'] = ma  

    ent = fast_glcm.fast_glcm_entropy(gray_image)
    ent = ent.reshape(-1)
    df['glcm_ent'] = ent  

    return df


def feature_extraction(image):
    df = pd.DataFrame()
 
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    df['gray'] = gray_image.reshape(-1)  

    color_feats = color_features(image)
    df = pd.concat([df, color_feats], axis=1)

    #HOG
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    df['hog'] = hog_image.reshape(-1) 

    #Texture feature extraction
    entropy_img = entropy(gray_image, disk(3))
    df['entropy'] = entropy_img.reshape(-1) 

    #Canny edge
    edges = cv.Canny(gray_image, 100,200)   
    edges = edges.reshape(-1)
    df['canny_edge'] = edges 

    #Gaussian with sigma=3
    gaussian_img = nd.gaussian_filter(gray_image, sigma=3)
    gaussian_img = gaussian_img.reshape(-1)
    df['gaussian_s3'] = gaussian_img

    #Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(gray_image, sigma=7)
    gaussian_img2 = gaussian_img2.reshape(-1)
    df['gaussian_s7'] = gaussian_img2

    #Median with sigma=3
    median_img = nd.median_filter(gray_image, size=3)
    median_img = median_img.reshape(-1)
    df['median_s3'] = median_img

    #Variance with size=3
    variance_img = nd.generic_filter(gray_image, np.var, size=3)
    variance_img = variance_img.reshape(-1)
    df['variance_s3'] = variance_img  

    #Prewitt
    edge_prewitt = prewitt(gray_image)
    edge_prewitt = edge_prewitt.reshape(-1)
    df['prewitt'] = edge_prewitt

    #Roberts edge
    edge_roberts = roberts(gray_image)
    edge_roberts = edge_roberts.reshape(-1)
    df['roberts'] = edge_roberts

    #Scharr
    edge_scharr = scharr(gray_image)
    edge_scharr = edge_scharr.reshape(-1)
    df['scharr'] = edge_scharr

    #Sobel
    edge_sobel = sobel(gray_image)
    edge_sobel = edge_sobel.reshape(-1)
    df['sobel'] = edge_sobel

    #Gabor
    gabor_feats = gabor_features(gray_image)
    df = pd.concat([df, gabor_feats], axis=1)

    #GLCM
    glcm_feats = glcm_features(gray_image)
    df = pd.concat([df, glcm_feats], axis=1)

    #LBP
    numPoints = 24
    radius = 8 
    lbp = local_binary_pattern(gray_image, numPoints, radius, method="uniform")
    lbp = lbp.reshape(-1)
    df['lbp'] = lbp  

    return df

def get_features(image_path):

    #Obtain image features
    image_dataset = pd.DataFrame() 
    for file in sorted(os.listdir(image_path)):  

        image = cv.imread(image_path + file)

        print("file: ", file)
        df = feature_extraction(image)

        image_dataset = pd.concat([image_dataset, df], axis=0)

    return image_dataset