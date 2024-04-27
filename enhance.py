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

def linear_gray_level_function(channel_image,  mode="normal", map_range=(25, 255),):
    c, d = map_range

    a, b = cv.minMaxLoc(channel_image)[:2]
    k = (d - c) / (b - a)
    map_image = k * (channel_image - a) + c
    
    return map_image



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
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
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



def SSR(img, variance=[15, 80, 150]):
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
           
    elif type == "MSR":
        r_map = MSR(r_image)
        g_map = MSR(g_image)
        b_map = MSR(b_image)     
    elif type == "SSR":
        r_map = SSR(r_image)
        g_map = SSR(g_image)
        b_map = SSR(b_image)     
    elif type == "gamma_transform":
        r_map = gamma_transform(r_image)
        g_map = gamma_transform(g_image)
        b_map = gamma_transform(b_image)     
        
        
    
    enhanced_image = cv.merge((r_map, g_map, b_map))
    # save_path = img_path.replace(".jpg", f"{type}.jpg")
    # cv.imwrite(save_path, enhanced_image)
    return enhanced_image

# image_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark\Dog\2015_05487.jpg"
# img = cv2.imread(image_path)

# img_enhance = enhance(img, "log_transform")

# cv2.imshow("Original Image", img)

# # Hiển thị hình ảnh đã được tăng cường
# cv2.imshow("Enhanced Image", img_enhance)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
    

