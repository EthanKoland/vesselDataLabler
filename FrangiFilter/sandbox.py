
import numpy as np
import matplotlib
import cv2
from frangiFilter2D import FrangiFilter2D

from PIL import Image



image1 = Image.open("CSAngioImages/0035.jpg")
image1 = image1.convert("RGB")
mask1 = Image.open("finalImagesV2/0035_mask.jpg")

mask2 = Image.open("finalImagesV2/0036_mask.jpg")
mask3 = Image.open("finalImagesV2/0034_mask.jpg")


imageArray1 = np.array(image1)
mask1Array = np.array(mask1)
mask2Array = np.array(mask2)
mask3Array = np.array(mask3)

mask3Array = mask3Array >= 128
mask2Array = mask2Array >= 128
mask1Array = mask1Array >= 128

referenceBase = np.array([[0,0,0], [255, 255, 255]]) * (0.80)
refernceRed = np.array([[0,0,0], [255, 0, 0]]) * (0.25)
refernceBlue = np.array([[0,0,0], [0, 0, 255]]) * (0.25)

mask3Array = np.take(refernceRed, mask3Array, axis=0)
mask2Array = np.take(refernceBlue, mask2Array, axis=0)
mask1Array = np.take(referenceBase, mask1Array, axis=0)


finalImage = imageArray1 + mask3Array + mask2Array + mask1Array
np.clip(finalImage, 0, 255, out=finalImage)

finalImage = Image.fromarray(finalImage.astype(np.uint8))
finalImage.show()
pass



