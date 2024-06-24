import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import os
from cv2 import imread, createCLAHE
import cv2
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image

from keras.models import load_model

from IPython.display import clear_output
from keras.optimizers import Adam


from keras.models import *
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input
from keras.optimizers import *
from keras import backend as keras




def getData(X_shape, image_path, training_files, flag = "test"):
    im_array = []
      
    if flag == "train":
        for i in tqdm(training_files):
            im = cv2.resize(cv2.imread(os.path.join(image_path,i)),(X_shape,X_shape))[:,:,0]
            im_array.append(im)

        return im_array

        
# Load training and testing data


def loadData(image_path = "./InprogressImages"):
    
    
# find the intersection between set(A) and set(B). both folders has files with the same name
    training_files = set(os.listdir(image_path))
    print(training_files)
    print("The number of training files is :",len(training_files))
    dim = 256*2
    X_train = getData(dim,image_path, training_files, flag="train")

    print("X_train[0] dims", X_train[0].shape)

    images = np.array(X_train).reshape(len(X_train),dim,dim,1)

    print("images dims", images.shape)
    
    return images
    




def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
    

# model = unet(input_size=(512,512,1))
# model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss,
#                   metrics=[dice_coef, 'binary_accuracy'])
# model.summary()

# Load weights of the model
# model.load_weights("./Cath_weights.best.hdf5")

def loadModel(weights_path = "./FrangiFilter/Cath_weights.best.hdf5"):
    model = unet(input_size=(512,512,1))
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss,
                  metrics=[dice_coef, 'binary_accuracy'])
    model.load_weights(weights_path)
    return model

def predictModel(model, images):
    train_vol = (images-127.0)/127.0
    preds = model.predict(train_vol)
    return preds



# train_vol = (images-127.0)/127.0
# print("train_vol dims", train_vol.shape)

# preds = model.predict(train_vol)

def displayPredictions(train_vol, preds):
    plt.figure(figsize=(20,10))

    for i in range(0,len(train_vol),10):
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(train_vol[i]))
        plt.xlabel("Base Image")

        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(preds[i]))
        plt.xlabel("Prediction")

        plt.show()
        
def savePredictions(preds, folder = "AI_Predictions"):
    

    
    for i in range(len(preds)):
        
        t0 = preds[i]
    
        t0 = np.round(t0)
        t0 = np.multiply(t0, 255)
        
        im = Image.fromarray(np.squeeze(t0).astype(np.uint8))
        im.convert("RGB")
        outputString = os.path.join(folder, f"prediction_{i}.jpg")
        im.save(outputString)
        
    print("Predictions saved")

# plt.figure(figsize=(20,10))

# for i in range(0,10,1):
#     plt.subplot(1,3,1)
#     plt.imshow(np.squeeze(train_vol[i]))
#     plt.xlabel("Base Image")
        
#     plt.subplot(1,3,3)
#     plt.imshow(np.squeeze(preds[i]))
#     plt.xlabel("Prediction")
    
#     plt.show()

if(__name__ == "__main__"):
    images = loadData()
    model = loadModel()
    preds = predictModel(model, images)
    savePredictions(preds)
    displayPredictions(images, preds)





