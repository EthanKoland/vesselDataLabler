import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import convolve
from frangiFilter2D import eig2image,FrangiFilter2D
from skimage.filters import threshold_otsu, try_all_threshold

img = cv2.imread("FrangiFilter/CS.bmp")
# img = cv2.imread("echo054.jpg")
# plt.imshow(img)
# plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.


Sigma=0.5
# Make kernel coordinates
X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

# Build the gaussian 2nd derivatives filters
DGaussxx = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussxy = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussyy = DGaussxx.conj().T

I=gray
Dxx = convolve(I, DGaussxx, mode='constant', cval=0.0)
DxxThreshold = Dxx > threshold_otsu(Dxx)
Dxy = convolve(I, DGaussxy, mode='constant', cval=0.0)
DxyThreshold = Dxy > threshold_otsu(Dxy)
Dyy = convolve(I, DGaussyy, mode='constant', cval=0.0)
DyyThreshold = Dyy > threshold_otsu(Dyy)
filterImg = FrangiFilter2D(gray)

# Calculate (abs sorted) eigenvalues and vectors
# Lambda2, Lambda1, Ix, Iy = eig2image(Dxx, Dxy, Dyy)

#Used to compare different methods of thresholding
# fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
# plt.show()

plt.subplot(231), plt.imshow(img), plt.axis('off'), plt.title('IMG', size=20), 
plt.subplot(232), plt.imshow(DxxThreshold*255, cmap='Greys'), plt.axis('off'), plt.title('DXX', size=20), 
plt.subplot(233), plt.imshow(DxyThreshold*255, cmap='Greys'), plt.axis('off'), plt.title('DXY', size=20)
plt.subplot(234), plt.imshow(DyyThreshold*255, cmap='Greys'), plt.axis('off'), plt.title('DYY', size=20)
plt.subplot(235), plt.imshow(filterImg[0], cmap='Greys'), plt.axis('off'), plt.title('Frangi', size=20)
plt.show()



