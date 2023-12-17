import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

Sigma=100
# Make kernel coordinates
X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

# Build the gaussian 2nd derivatives filters
DGaussxx = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussxy = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussyy = DGaussxx.conj().T

# print(DGaussxx.shape)
# print(DGaussxx)
print(DGaussxy)

plt.imshow(DGaussxx, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(DGaussxy, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(DGaussyy, cmap='hot', interpolation='nearest')
plt.show()