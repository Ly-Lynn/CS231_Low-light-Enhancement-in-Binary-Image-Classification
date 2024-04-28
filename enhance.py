# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def linearTransformed(image,c, d):
#     # Apply linear gray level transformation
#     a = np.min(image)
#     b = np.max(image)
#     k = (d - c) / (b - a)
#     print(f"a={a}, b={b}, c={c}, d={d}")
#     print(f"k={k}")
#     transformed_image = (image - a) * k + c
#     transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
#     return transformed_image

# def logarithmTransformed(image, ld = 70, lg=10):
#     transformed_image = ld * np.log10(1 + image) 
#     transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
#     return transformed_image
    
# def gammaTransformed(image, ld = 25, ep=1, gam = 0.5):
#     transformed_image = ld * (image + ep)**gam 
#     transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
#     return transformed_image


# img_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\image_test\2015_05025.jpg"
# img = cv2.imread(img_path)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = logarithmTransformed(img)
# cv2.imwrite(img_path.replace(".jpg", "_enhaced.jpg"), img)

import cv2 as cv
import numpy as np
import os
import json
from PIL import Image


def linear_gray_level_function(channel_image,  mode="normal", map_range=(25, 255),):
    c, d = map_range

    a, b = cv.minMaxLoc(channel_image)[:2]
    k = (d - c) / (b - a)
    map_image = k * (channel_image - a) + c
    
    return map_image.astype(np.uint8)



def log_transform(channel, lamda=1, v=100):
    channel = channel.astype(np.float32) / 255.0
    
    transformed_channel = lamda * np.log(1 + v * channel) / np.log(1 + v)
    
    transformed_channel = (transformed_channel * 255).astype(np.uint8)
    
    return transformed_channel


def gamma_transform(channel, lamda=2, gramma=0.8, epsilon=0.5):
    channel = channel.astype(np.float32) / 255.0
    transformed_channel = lamda * (channel + epsilon) ** gramma
    transformed_channel = (transformed_channel * 255).astype(np.uint8)
    
    return transformed_channel



def singleScaleRetinex(img, variance=15):
    retinex = np.log10(img) - np.log10(cv.GaussianBlur(img, (0, 0), variance))
    return retinex

def multiScaleRetinex(img, variance_list=[15, 80, 150]):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex

   

def MSR(img, variance_list=[15, 80, 150]):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)
    zero_count = 0

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break   
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex



def SSR(img, variance=200):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    zero_count = 0

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex


    
    
def enhance(image, type):

    r_image = image[:, :, 0]
    g_image = image[:, :, 1]
    b_image = image[:, :, 2]
    
    if type == "linear_gray_transform":
        r_map = linear_gray_level_function(r_image)
        g_map = linear_gray_level_function(g_image)
        b_map = linear_gray_level_function(b_image)
       
    elif type == "log_transform":
        r_map = log_transform(r_image)
        g_map = log_transform(g_image)
        b_map = log_transform(b_image)
    elif type == "gamma_transform":
        r_map = gamma_transform(r_image)
        g_map = gamma_transform(g_image)
        b_map = gamma_transform(b_image)  
        
        
    enhanced_image = cv.merge((r_map, g_map, b_map))
    # save_path = img_path.replace(".jpg", f"{type}.jpg")
    # cv.imwrite(save_path, enhanced_image)
    return enhanced_image

def save_enhance(annotator_file,
                enhance_type,
                mode="Train",
                anno_dir = r"D:/AI/CV/CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks/ExDark_Annno",
                img_dir = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark"):
    
    OUTDIR = f"Enhance_{enhance_type}"
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
        
    MODE_OUTDIR = os.path.join(OUTDIR, mode)
    if not os.path.exists(MODE_OUTDIR):
        os.mkdir(MODE_OUTDIR)
    
    with open(annotator_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        
        
    for line in lines:
        [anno_path, label] = line.split(", ")
        img_path = anno_path.replace(anno_dir, img_dir).replace(".txt", "")
        img = cv.imread(img_path, cv.COLOR_BGR2RGB)
        file_name = f"{label}_{img_path[-14:]}"
        
        enhanced_img = enhance(img, enhance_type)
        combined_img = cv.hconcat([img, enhanced_img])
        cv.imwrite(os.path.join(MODE_OUTDIR, file_name), combined_img)
            
    