import cv2
import numpy as np
import matplotlib.pyplot as plt

def linearTransformed(image,c, d):
    # Apply linear gray level transformation
    a = np.min(image)
    b = np.max(image)
    k = (d - c) / (b - a)
    print(f"a={a}, b={b}, c={c}, d={d}")
    print(f"k={k}")
    transformed_image = (image - a) * k + c
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image

def logarithmTransformed(image, ld = 70, lg=10):
    transformed_image = ld * np.log10(1 + image) 
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image
    
def gammaTransformed(image, ld = 25, ep=1, gam = 0.5):
    transformed_image = ld * (image + ep)**gam 
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image


img_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\image_test\2015_05025.jpg"
img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = logarithmTransformed(img)
cv2.imwrite(img_path.replace(".jpg", "_enhaced.jpg"), img)

