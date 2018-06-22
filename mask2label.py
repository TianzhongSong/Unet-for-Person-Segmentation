import cv2
import numpy as np

im = cv2.imread('mask.jpg')
label = np.zeros_like(im, dtype='uint8')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, im = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        if im[i, j] != 0:
            label[i, j] = 1
cv2.imwrite('label.png', label)
