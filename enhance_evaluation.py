import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_lib
from skimage.metrics import mean_squared_error as mse 

def MSE(image, enhanced_image, image_size):
    '''
    Mean Square Error: 
    Represents the direct deviation between the enhanced image and the original image.
    The smaller the MSE, the higher the similarity between the enhanced and the original.    
    '''
    image = cv2.resize(image, image_size)
    enhanced_image = cv2.resize(enhanced_image, image_size)
    M, N = image.shape[0], image.shape[1]
    mse = np.sum((image - enhanced_image) ** 2) / (M * N)
    return mse

def PSNR(mse, max_pixel=255):
    '''
    Peak Signal-to-Noise Ratio: 
    Measure of how much noise is present in the image relative to the maximum possible noise level.
    The larger the PSNR, the smaller the differences between the enhanced and the original.
    Intensively low PSNR can suggest that the enhanced image is servely noised.
    '''
    if mse == 0: 
        return float('inf')
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def ssim(image, enhance_image):
    '''
    A perception-based method measuring similarity of 2 images. 
    Including: 
        l(x,y): luminance (intensity) comparison, 
        c(x,y): contrast comparison,
        s(x,y): structure comparison 
    SSIM(x,y) = l(x,y) . c(x,y) . s(x,y)
    '''
    # constant variable stabilize the division when closing to 0
    C1 = (0.01 * 255)**2 
    C2 = (0.03 * 255)**2

    image = image.astype(np.float64)
    enhance_image = enhance_image.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(image, -1, window)[5:-5, 5:-5] # mean value of original image
    mu2 = cv2.filter2D(enhance_image, -1, window)[5:-5, 5:-5] # mean value of enhance image
    mu1_sq = mu1**2 
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # variance of local regions: sigma^2 = E[X^2] - (E[X])^2
    sigma1_sq = cv2.filter2D(image**2, -1, window)[5:-5, 5:-5] - mu1_sq 
    sigma2_sq = cv2.filter2D(enhance_image**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(image * enhance_image, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def SSIM(image, enhanced_image, size):
    image = cv2.resize(image, size)
    enhanced_image = cv2.resize(enhanced_image, size)
    if image.ndim == 2:
        return ssim(image, enhanced_image)
    elif image.ndim == 3:
        if image.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(image, enhanced_image))
            return np.array(ssims).mean()
        elif image.shape[2] == 1:
            return ssim(np.squeeze(image), np.squeeze(enhanced_image))