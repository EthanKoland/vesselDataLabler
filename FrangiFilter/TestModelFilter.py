import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout,concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Reshape,
    Flatten,
    Dense,
)

IMAGE_SIZE = 200

Sigma=2
# Make kernel coordinates
X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

# Build the gaussian 2nd derivatives filters
DGaussxx = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussxy = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussyy = DGaussxx.conj().T

print(DGaussxx.shape)
# Test codes for combining filters
# ff = DGaussxx[...,np.newaxis,np.newaxis]
# ff2 = DGaussxy[...,np.newaxis,np.newaxis]
# print(ff.shape)
# ff = np.append(ff,ff2,axis=3)
# print(ff.shape)

# custom filter
def my_filter(shape, dtype=None):
    f = DGaussxx[...,np.newaxis,np.newaxis]
    f2 = DGaussxy[...,np.newaxis,np.newaxis]
    f = np.append(f,f2,axis=3)
    f3 = DGaussyy[...,np.newaxis,np.newaxis]
    f = np.append(f,f3,axis=3)
    # assert f.shape == shape
    return K.variable(f, dtype='float32')

def generate_model():
    input_tensor = Input(shape=(200,200,1))
    filterout = Conv2D(filters=3, 
                      kernel_size = 13,
                      kernel_initializer=my_filter,
                      padding='same') (input_tensor)
    randout = Conv2D(5,kernel_size=3, use_bias=True, padding="same", activation="relu")(input_tensor)
    merged = concatenate([filterout, randout], axis=3)

    model = Model(inputs=input_tensor, outputs=merged)
    return model

# model
model = generate_model()
model.summary()

img = cv2.imread(f"echo112.jpg")
img = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fimg = gray/ 255.0
plt.imshow(img)
plt.show()

# reshape(1 (first one is the number of images, 200 (image width), 200 (image height), 1 (one color channel))
in_img = np.array(fimg).reshape(1,200,200,1)
preds = np.asarray(model.predict(in_img))

print(preds.shape)
img=np.squeeze(preds[0])
print(img.shape)

# First channel for filter DGaussxx
plt.imshow(img[:,:,0])
plt.show()

# second channel for filter DGaussxy
plt.imshow(img[:,:,1])
plt.show()

plt.imshow(img[:,:,2])
plt.show()