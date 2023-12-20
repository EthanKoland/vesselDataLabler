import numpy as np
import matplotlib
import cv2
from frangiFilter2D import FrangiFilter2D

# diameter = 9
# SqDist = (diameter/2)**2
# sddist2 = (diameter//2)**2
# sq = np.zeros((diameter, diameter))
# sq2 = np.zeros((diameter, diameter))
# for i in range(diameter):
#     for j in range(diameter):
#         sq[i][j] = ((i-diameter//2)**2 + (j-diameter//2)**2)>SqDist
#         sq2[i][j] = ((i-diameter//2)**2 + (j-diameter//2)**2)>sddist2
        
# print(sq)
# print(sq2)

kwargs ={}

FrangiScaleRange = kwargs.get("FrangiScaleRange", (1,10))
FrangiScaleRatio = kwargs.get("FrangiScaleRatio", 2)
FrangiBetaOne = kwargs.get("FrangiBetaOne", 0.5)
FrangiBetaTwo = kwargs.get("FrangiBetaTwo", 15)
verbose = kwargs.get("verbose", False)
BlackWhite = kwargs.get("BlackWhite", True)
filterlocation = kwargs.get("filterloaction", "filterMasks")

image = cv2.imread("CSAngioImages/0000.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.

mask = FrangiFilter2D(gray,FrangiScaleRange=FrangiScaleRange, FrangiScaleRatio=FrangiScaleRatio,
                        FrangiBetaOne=FrangiBetaOne, FrangiBetaTwo=FrangiBetaTwo,
                        verbose=verbose, BlackWhite=BlackWhite)

grayout = cv2.normalize(mask[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
ret2,th2 = cv2.threshold(grayout,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# manual threshold

# ret2,th2 = cv2.threshold(grayout,8,255,cv2.THRESH_BINARY)

m2 = th2 == 255

m3 = m2 * 255

print(np.unique(th2))
print(np.unique(m3))
