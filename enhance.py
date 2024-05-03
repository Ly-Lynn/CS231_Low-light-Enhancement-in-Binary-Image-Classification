import cv2 as cv
import numpy as np
import os
import json
from PIL import Image
from keras import layers
import tensorflow as tf
from keras.layers import add, Dense, Dropout, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample

def up(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:        
#       upsample.dropout(0.2)
        upsample.add(Dropout(0.1))
    upsample.add(keras.layers.LeakyReLU())
    return upsample

def Autoencoder(SIZE=256):
    inputs = layers.Input(shape= [SIZE,SIZE,3])
    d1 = down(128,(3,3),False)(inputs)
    d2 = down(128,(3,3),False)(d1)
    d3 = down(256,(3,3),True)(d2)
    d4 = down(512,(3,3),True)(d3)
    
    d5 = down(512,(3,3),True)(d4)

    u1 = up(512,(3,3),True)(d5)
    u1 = layers.concatenate([u1,d4])
    u2 = up(256,(3,3),True)(u1)
    u2 = layers.concatenate([u2,d3])
    u3 = up(128,(3,3),True)(u2)
    u3 = layers.concatenate([u3,d2])
    u4 = up(128,(3,3),True)(u3)
    u4 = layers.concatenate([u4,d1])
    u5 = up(3,(3,3),True)(u4)
    u5 = layers.concatenate([u5,inputs])
    output = layers.Conv2D(3,(2,2),strides = 1, padding = 'same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)


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

def histogram_equalization(image):
    r, g, b = cv.split(image)
    b_equalized = cv.equalizeHist(b)
    g_equalized = cv.equalizeHist(g)
    r_equalized = cv.equalizeHist(r)
    equalized_image = cv.merge((r_equalized, g_equalized, b_equalized))

    return equalized_image

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


    
    
def enhance(image, type, model=None):
    if not model:
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
    elif type == "HE":
        return histogram_equalization(image)
   
    elif type == "Autoencoder":
        img = cv.resize(image, (256, 256)) 
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img_pred = model.predict(img)
        return np.clip(img_pred, 0.0, 1.0)[0] * 255.0
        
    enhanced_image = cv.merge((r_map, g_map, b_map))
    # save_path = img_path.replace(".jpg", f"{type}.jpg")
    # cv.imwrite(save_path, enhanced_image)
    return enhanced_image

def save_enhance(annotator_file,
                enhance_type,
                mode="Train",
                model=None,
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

        
        enhanced_img = enhance(img, enhance_type, model)
        if model:
            img = cv.resize(img, (256, 256)).astype('float32')
        combined_img = cv.hconcat([img, enhanced_img])
        cv.imwrite(os.path.join(MODE_OUTDIR, file_name), combined_img)


       
# save_enhance("Splits/Train.txt",
#              "Autoencoder",
#              "Train",
#              model)
# save_enhance("Splits/Test.txt",
#              "Autoencoder",
#              "Test",
#              model)

# img_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark\Dog\2015_04963.jpg"


# img = cv.imread(img_path, cv.COLOR_BGR2RGB)
# enhanced = enhance(img, "Autoencoder", model)
# # img = cv.resize(img, (256, 256)) 
# # img = img.astype('float32') / 255.0
# # img = np.expand_dims(img, axis=0)
# # img_pred = model.predict(img)[0]
# # img_pred *= 255
# print(enhanced)
# cv.imshow("prediction: ",enhanced)
# cv.imwrite("test.png", enhanced * 255)
# # img_pred_pil = Image.fromarray(enhanced)
# # img_pred_pil.save("test.png")

# cv.waitKey(0)
# cv.destroyAllWindows()