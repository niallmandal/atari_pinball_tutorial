import noise
import cv2
import numpy as np

img = cv2.imread('green.jpg')
img = np.uint8(noise.noisy(img, 0.5))

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.desAllWindows()
