import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import cv2
from PIL import Image
from frangiFilter2D import eig2image,FrangiFilter2D
from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_otsu, try_all_threshold

def maskTransparency():
    
    convertID = lambda x: "0" * (4 - len(str(x))) + str(x) if(len(str(x)) < 4) else str(x)
    modifiedMask = lambda mask, alpha: mask/255 * alpha
    
    imageID = 35
    imgRaw = imread(f"CSAngioImages/{convertID(imageID)}.jpg")
    maskRaw = imread(f"finalImagesV2/{convertID(imageID)}_mask.jpg")
    
    cmapImg = "gray"
    cmapMask = "plasma"
    
    #Create a 3,5 subplot
    fig, ax = plt.subplots(2,3)
    
    ax[0,0].imshow(imgRaw, cmap=cmapImg)
    ax[0,0].set_title("0% Transparency")
    
    ax[0,1].imshow(imgRaw, cmap=cmapImg)
    ax[0,1].imshow(maskRaw, alpha=modifiedMask(maskRaw, 0.1), cmap=cmapMask, )
    ax[0,1].set_title("20% Transparency")
    
    ax[0,2].imshow(imgRaw, cmap = cmapImg)
    ax[0,2].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[0,2].set_title("40% Transparency")
    
    ax[1,0].imshow(imgRaw, cmap = cmapImg)
    ax[1,0].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1,0].set_title("60% Transparency")
    
    ax[1,1].imshow(imgRaw, cmap = cmapImg)
    ax[1,1].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1,1].set_title("80% Transparency")
    
    ax[1,2].imshow(imgRaw, cmap = cmapImg)
    ax[1,2].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1,2].set_title("100% Transparency")
    
    
    plt.show()
    
def isolatedPlt():
    convertID = lambda x: "0" * (4 - len(str(x))) + str(x) if(len(str(x)) < 4) else str(x)
    modifiedMask = lambda mask, alpha: mask/255 * alpha
    
    imageID = 35
    imgRaw = imread(f"CSAngioImages/{convertID(imageID)}.jpg")
    maskRaw = imread(f"finalImagesV2/{convertID(imageID)}_mask.jpg")
    
    cmapImg = "plasma"
    cmapMask = "gray"
    
    #Create a 3,5 subplot
    fig, ax = plt.subplots(1,2)
    

    
    ax[0].imshow(imgRaw, cmap = cmapImg)
    ax[0].imshow(maskRaw, alpha=modifiedMask(maskRaw, 0.5), cmap=cmapMask)
    ax[0].set_title("80% Transparency")
    
    ax[1].imshow(imgRaw, cmap = cmapImg)
    ax[1].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1].set_title("100% Transparency")
    
    plt.show()
    
def surrondingComparision():
    convertID = lambda x: "0" * (4 - len(str(x))) + str(x) if(len(str(x)) < 4) else str(x)
    modifiedMask = lambda mask, alpha: mask/255 * alpha
    
    imageID = 35
    imgRaw = imread(f"CSAngioImages/{convertID(imageID)}.jpg")
    maskRaw = imread(f"finalImagesV2/{convertID(imageID)}_mask.jpg")
    nextMask = imread(f"finalImagesV2/{convertID(imageID+1)}_mask.jpg")
    prevMask = imread(f"finalImagesV2/{convertID(imageID-1)}_mask.jpg")
    
    cmapImg = "gray_r"
    cmapMask = "gray"
    cmapNext = "Reds"
    cmapReverse = "Blues"
    
    alphaRaw = modifiedMask(maskRaw, 0)

    #Create a 3,5 subplot
    fig, ax = plt.subplots(4,5)
    
    ax[0,0].imshow(imgRaw, cmap=cmapImg)
    ax[0,0].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[0,0].set_title("20% Transparency")
    ax[0,0].axis('off')
    
    ax[0,1].imshow(imgRaw, cmap=cmapImg)
    ax[0,1].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[0,1].set_title("40% Transparency")
    ax[0,1].axis('off')
    
    ax[0,2].imshow(imgRaw, cmap=cmapImg)
    ax[0,2].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[0,2].set_title("60% Transparency")
    ax[0,2].axis('off')
    
    ax[0,3].imshow(imgRaw, cmap=cmapImg)
    ax[0,3].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[0,3].set_title("80% Transparency")
    ax[0,3].axis('off')
    
    ax[0,4].imshow(imgRaw, cmap=cmapImg)
    ax[0,4].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[0,4].set_title("100% Transparency")
    ax[0,4].axis('off')
    
    ax[1,0].imshow(imgRaw, cmap=cmapImg)
    ax[1,0].imshow(nextMask, alpha=modifiedMask(nextMask, 0.1), cmap=cmapNext)
    ax[1,0].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1,0].axis('off')
    
    ax[1,1].imshow(imgRaw, cmap=cmapImg)
    ax[1,1].imshow(nextMask, alpha=modifiedMask(nextMask, 0.2), cmap=cmapNext)
    ax[1,1].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1,1].axis('off')
    
    ax[1,2].imshow(imgRaw, cmap=cmapImg)
    ax[1,2].imshow(nextMask, alpha=modifiedMask(nextMask, 0.3), cmap=cmapNext)
    ax[1,2].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1,2].axis('off')
    
    ax[1,3].imshow(imgRaw, cmap=cmapImg)
    ax[1,3].imshow(nextMask, alpha=modifiedMask(nextMask, 0.4), cmap=cmapNext)
    ax[1,3].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1,3].axis('off')
    
    ax[1,4].imshow(imgRaw, cmap=cmapImg)
    ax[1,4].imshow(nextMask, alpha=modifiedMask(nextMask, 0.5), cmap=cmapNext)
    ax[1,4].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[1,4].axis('off')
    
    ax[2,0].imshow(imgRaw, cmap=cmapImg)
    ax[2,0].imshow(prevMask, alpha=modifiedMask(prevMask, 0.1), cmap=cmapReverse)
    ax[2,0].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[2,0].axis('off')
    
    ax[2,1].imshow(imgRaw, cmap=cmapImg)
    ax[2,1].imshow(prevMask, alpha=modifiedMask(prevMask, 0.2), cmap=cmapReverse)
    ax[2,1].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[2,1].axis('off')
    
    ax[2,2].imshow(imgRaw, cmap=cmapImg)
    ax[2,2].imshow(prevMask, alpha=modifiedMask(prevMask, 0.3), cmap=cmapReverse)
    ax[2,2].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[2,2].axis('off')
    
    ax[2,3].imshow(imgRaw, cmap=cmapImg)
    ax[2,3].imshow(prevMask, alpha=modifiedMask(prevMask, 0.4), cmap=cmapReverse)
    ax[2,3].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[2,3].axis('off')
    
    ax[2,4].imshow(imgRaw, cmap=cmapImg)
    ax[2,4].imshow(prevMask, alpha=modifiedMask(prevMask, 0.5), cmap=cmapReverse)
    ax[2,4].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[2,4].axis('off')
    
    ax[3,0].imshow(imgRaw, cmap=cmapImg)
    ax[3,0].imshow(nextMask, alpha=modifiedMask(nextMask, 0.1), cmap=cmapNext)
    ax[3,0].imshow(prevMask, alpha=modifiedMask(prevMask, 0.1), cmap=cmapReverse)
    ax[3,0].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[3,0].axis('off')
    
    ax[3,1].imshow(imgRaw, cmap=cmapImg)
    ax[3,1].imshow(nextMask, alpha=modifiedMask(nextMask, 0.2), cmap=cmapNext)
    ax[3,1].imshow(prevMask, alpha=modifiedMask(prevMask, 0.2), cmap=cmapReverse)
    ax[3,1].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[3,1].axis('off')
    
    ax[3,2].imshow(imgRaw, cmap=cmapImg)
    ax[3,2].imshow(nextMask, alpha=modifiedMask(nextMask, 0.3), cmap=cmapNext)
    ax[3,2].imshow(prevMask, alpha=modifiedMask(prevMask, 0.3), cmap=cmapReverse)
    ax[3,2].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[3,2].axis('off')
    
    ax[3,3].imshow(imgRaw, cmap=cmapImg)
    ax[3,3].imshow(nextMask, alpha=modifiedMask(nextMask, 0.4), cmap=cmapNext)
    ax[3,3].imshow(prevMask, alpha=modifiedMask(prevMask, 0.4), cmap=cmapReverse)
    ax[3,3].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[3,3].axis('off')
    
    ax[3,4].imshow(imgRaw, cmap=cmapImg)
    ax[3,4].imshow(nextMask, alpha=modifiedMask(nextMask, 0.5), cmap=cmapNext)
    ax[3,4].imshow(prevMask, alpha=modifiedMask(prevMask, 0.5), cmap=cmapReverse)
    ax[3,4].imshow(maskRaw, alpha=alphaRaw, cmap=cmapMask)
    ax[3,4].axis('off')
    
    plt.savefig("maskOverlap.png", dpi=1200)
    
    
def isolateMask(mask):
    mask = mask.copy()
    
    mask[mask != 0] = 1
    
    return mask
    
    
    
if(__name__ == "__main__"):
    #maskTransparency()
    surrondingComparision()
    
    



