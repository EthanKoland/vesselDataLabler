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

def initTest():
    img = imread("FrangiFilter/CS.bmp")
    plt.imshow(img)
    plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.

    # Apply Frangi filter (wire/vessel enhancement filter)
    filterImg = FrangiFilter2D(gray,FrangiScaleRange=np.array([1, 10]), FrangiScaleRatio=2, FrangiBetaOne=0.8, FrangiBetaTwo=15)
    # filterImg[0] is maximum values; filterImg[1] is scale values; filterImg[2] is angle values. We just use the maximum values
    grayout = cv2.normalize(filterImg[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(grayout,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Convert one channel image to three-channel image, so plt.imshow shows correct colors.
    threeout = cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)

    plt.imshow(threeout)
    plt.show()
    
def testScaleRange(FrangiScaleRange, FrangiScaleRatio=2, FrangiBetaOne=0.8, FrangiBetaTwo=15, col = 3, row = 3):
    img = cv2.imread("FrangiFilter/CS.bmp")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.
    
    scales = []
    
    for i in FrangiScaleRange:
        scales.append(FrangiFilter2D(gray,FrangiScaleRange=[i], FrangiScaleRatio=1, FrangiBetaOne=FrangiBetaOne, FrangiBetaTwo=FrangiBetaTwo))
        
    # pltUnified(gray, scales, FrangiScaleRange)
    pltSeperate(gray, scales, col, row)
    
def testFrangiBetaOne(FrangiBetaOneValues,FrangiScaleRange = np.arange(1,10), FrangiScaleRatio=2, FrangiBetaTwo=15):
    img = cv2.imread("FrangiFilter/CS.bmp")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.
    
    scales = []
    
    for i in FrangiBetaOneValues:
        scales.append(FrangiFilter2D(gray,FrangiScaleRange=FrangiScaleRange, FrangiScaleRatio=1, FrangiBetaOne=i, FrangiBetaTwo=FrangiBetaTwo))
        
    pltUnified(gray, scales, FrangiBetaOneValues)
    
def testFrangiBetaTwo(FrangiBetaTwoValues,FrangiScaleRange = np.arange(1,10), FrangiScaleRatio=2, FrangiBetaOne = 0.8):
    img = cv2.imread("FrangiFilter/CS.bmp")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.
    
    scales = []
    
    for i in FrangiBetaTwoValues:
        scales.append(FrangiFilter2D(gray,FrangiScaleRange=FrangiScaleRange, FrangiScaleRatio=1, FrangiBetaOne=FrangiBetaOne, FrangiBetaTwo=i))
        
    pltUnified(gray, scales, FrangiBetaTwoValues)
        
    
    
def pltSeperate(img, scales, col = 3, row = 3):
    fig, axs = plt.subplots(row, col)
    
    count = 0
    for i in range(row):
        for j in range(col):
            # t = scales[i*row+j].reshape(scales[i*row+j].shape[0], scales[i*row+j].shape[1])
            if(count >= len(scales)):
                break
            
            axs[i, j].imshow(scales[i*col+j][0], cmap='Greys_r')
            axs[i, j].axis('off')
            axs[i, j].set_title('Scale: '+str(i*col+j))
            count += 1 
      
    plt.show()
    
def pltUnified(img, scales, legend):
    fig, axs = plt.subplots(2, 2)
    
    colors = np.linspace(0, 1, len(scales))
    cmap = colormaps['rainbow']
    colors = cmap(colors)
    
    counts = []
    
    axs[0][0].imshow(img, cmap='Greys')
    axs[0][0].axis('off')
    axs[0][0].set_title('Raw Image')
    
    blankImage = np.zeros((img.shape[0], img.shape[1], 3))
    img2 = img.copy()
    img2 = gray2rgb(img2)
    for i,s in enumerate(scales[::-1]):
        th2 = s[0] > threshold_otsu(s[0], nbins=256)
        counts.append(np.count_nonzero(th2))
        for r in range(th2.shape[0]):
            for c in range(th2.shape[1]):
                if(th2[r][c] == True):
                    blankImage[r][c] = colors[i][:3]
                    img2[r][c] = colors[i][:3]
        
    axs[0][1].imshow(blankImage)
    axs[0][1].axis('off')
    axs[0][1].set_title('Frangi Filtered Masks')
    
    axs[1][0].imshow(img2)
    axs[1][0].axis('off')
    axs[1][0].set_title('Frangi Filtered Image')
    
    for i,c in enumerate(colors):
        axs[1][1].scatter(i + 1, counts[i], color=c, label=legend[i])
    
    axs[1][1].set_title('Number of Pixels in Masks')
    axs[1][1].set_xlabel('Scale')
    axs[1][1].set_ylabel('Number of Pixels')
    axs[1][1].legend()
    
        
    plt.show()
    
    
if(__name__ == "__main__"):
    testScaleRange(np.arange(1, 10))
    t = np.round(np.arange(0.1, 1, 0.1), 1)
    # testFrangiBetaOne(t)
    t2 = np.arange(1, 20, 2)
    # testFrangiBetaTwo(t2, FrangiBetaOne=1)
    
    
    



