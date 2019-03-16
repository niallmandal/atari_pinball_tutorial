import numpy as np
import cv2

def noisy(image,factor):
    row, col, ch =image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)
    noisy = image + image * gauss * factor
    return noisy
