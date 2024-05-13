import numpy as np

def MSE(image, enhanced_image):
    '''
    Mean Square Error: 
    Represents the direct deviation between the enhanced image and the original image.
    The smaller the MSE, the higher the similarity between the enhanced and the original.    
    '''
    M, N = image.shape
    mse = np.sum((image - enhanced_image) ** 2) / (M * N)
    return mse

def PSNR(mse, max_pixel=255):
    '''
    Peak Signal-to-Noise Ratio: 
    Measure of how much noise is present in the image relative to the maximum possible noise level.
    The larger the PSNR, the smaller the differences between the enhanced and the original.
    Intensively low PSNR can suggest that the enhanced image is servely noised.
    '''
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

