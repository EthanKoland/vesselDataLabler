import cv2
import numpy as np

from matplotlib import pyplot as plt

sobel_kernel_size = 3

img = cv2.imread("FrangiFilter/0002.jpg", cv2.IMREAD_GRAYSCALE)
canny =     cv2.Canny(img, 100, 200)
sobelX =    cv2.Sobel(img, cv2.CV_64F,1,0,ksize=sobel_kernel_size)
sobelY =     cv2.Sobel(img, cv2.CV_64F,0,1,ksize=sobel_kernel_size)
lap = cv2.Laplacian(img, cv2.CV_64F,ksize=3)
lap = np.uint8(np.absolute(lap))

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)
edge_mask = np.where(sobelCombined > 46,255, 0)


titles = ['Andiogram','canny','sobelX','sobelY','sobelCombined','edge_mask']
images = [img,canny,sobelX,sobelY,sobelCombined,edge_mask]

plt.figure(figsize=(10, 6))
for i in range (6):
    plt.subplot(2,3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles [i])
    plt.xticks([]),plt.yticks ([])

plt.show()