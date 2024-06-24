import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import convolve
from frangiFilter2D import eig2image,FrangiFilter2D
from skimage.filters import threshold_otsu, try_all_threshold

def kernalSize(values = [1,2,3,4], function = cv2.MORPH_OPEN, iterations = 1, filePath = "CSAngioImages/0032.jpg"):

    img = cv2.imread(filePath)
    # img = cv2.imread("echo054.jpg")
    # plt.imshow(img)
    # plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.
    # gray = cv2.subtract(1, gray)
    # gray = cv2.absdiff(gray, 1)
    #inverted_
    # gray = cv2.bitwise_not(gray)


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
    
    combined = DyyThreshold.astype(np.uint8) + DxxThreshold.astype(np.uint8)
    np.minimum(combined, 1, combined)
    
    fig, axs = plt.subplots(4, len(values))
    
    for i in range(len(values)):
        value = values[i]
        
        # Opening
        kernel = np.ones((value,value),np.uint8)
        DxxMorph = cv2.morphologyEx(DxxThreshold.astype(np.uint8), function, kernel, iterations=iterations)
        DxyMorph = cv2.morphologyEx(DxyThreshold.astype(np.uint8), function, kernel, iterations=iterations) 
        DyyMorph = cv2.morphologyEx(DyyThreshold.astype(np.uint8), function, kernel, iterations=iterations  )
        
        combinedMorph = cv2.morphologyEx(combined.astype(np.uint8), function, kernel, iterations=value)
        
        axs[0][i].imshow(DxxMorph, cmap='Greys')
        axs[0][i].axis('off')
        axs[0][i].set_title(f'DXX w/ opening{value}', size=5)
        
        axs[1][i].imshow(DxyMorph, cmap='Greys')
        axs[1][i].axis('off')
        axs[1][i].set_title(f'DXy w/ opening{value}', size=5)
        
        axs[2][i].imshow(DyyMorph, cmap='Greys')
        axs[2][i].axis('off')
        axs[2][i].set_title(f'DYY w/ opening{value}', size=5)
        
        axs[3][i].imshow(combinedMorph, cmap='Greys')
        axs[3][i].axis('off')
        axs[3][i].set_title(f'Combined w/ opening{value}', size=5)
        
    plt.show()

def iterationsOpening(values = [1,2,3,4], function = cv2.MORPH_OPEN, kernelSize = 3, filePath = "CSAngioImages/0032.jpg"):

    img = cv2.imread(filePath)
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
    
    combined = DyyThreshold.astype(np.uint8) + DxxThreshold.astype(np.uint8)
    np.minimum(combined, 1, combined)
    
    fig, axs = plt.subplots(4, len(values))
    
    for i in range(len(values)):
        value = values[i]
        
        # Opening
        kernel = np.ones((kernelSize,kernelSize),np.uint8)
        DxxMorph = cv2.morphologyEx(DxxThreshold.astype(np.uint8), function, kernel, iterations=value)
        DxyMorph = cv2.morphologyEx(DxyThreshold.astype(np.uint8), function, kernel, iterations=value)
        DyyMorph = cv2.morphologyEx(DyyThreshold.astype(np.uint8), function, kernel, iterations=value)
        combinedMorph = cv2.morphologyEx(combined.astype(np.uint8), function, kernel, iterations=value)
        
        axs[0][i].imshow(DxxMorph, cmap='Greys')
        axs[0][i].axis('off')
        axs[0][i].set_title(f'DXX w/ opening{value}', size=5)
        
        axs[1][i].imshow(DxyMorph, cmap='Greys')
        axs[1][i].axis('off')
        axs[1][i].set_title(f'DXy w/ opening{value}', size=5)
        
        axs[2][i].imshow(DyyMorph, cmap='Greys')
        axs[2][i].axis('off')
        axs[2][i].set_title(f'DYY w/ opening{value}', size=5)
        
        axs[3][i].imshow(combinedMorph, cmap='Greys')
        axs[3][i].axis('off')
        axs[3][i].set_title(f'Combined w/ opening{value}', size=5)
        
    plt.show()
    
def frangiOppeingClosing(currentImage, kernalSize = [1,2,3,4], iterations = [1,2,3,4], function = cv2.MORPH_OPEN ):
    image = cv2.imread(currentImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
    
    
    mask = FrangiFilter2D(gray)
    
    #Get the base name of the image file
    imageName = currentImage.split("/")[-1]
    imageName = imageName.split(".")[0]
    
    grayout = cv2.normalize(mask[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    # Otsu's thresholding

    ret2,th2 = cv2.threshold(grayout,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # manual threshold
    threeout = cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)
    
    fig, axs = plt.subplots(len(kernalSize), len(iterations))

    # ret2,th2 = cv2.threshold(grayout,8,255,cv2.THRESH_BINARY)
    
    for i,kSize in enumerate(kernalSize):
        for j,iter in enumerate(iterations):
            t = cv2.morphologyEx(threeout.astype(np.uint8), function, kSize)
            axs[i][j].imshow(th2, cmap='Greys')
            axs[i][j].axis('off')
            axs[i][j].set_title(f'kSize: {kSize}, iter: {iter}', size=5)
            
    plt.show()
            
            
            
    
    m2 = th2 == 255
    
if __name__ == "__main__":
   kernalSize([1,2,3,4], cv2.MORPH_CLOSE, 1)
    #iterationsOpening([1,2,3,4], kernelSize=2, function=cv2.MORPH_OPEN)
    # frangiOppeingClosing("CSAngioImages/0032.jpg")



