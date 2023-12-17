import matplotlib.pyplot as plt
import numpy as np  

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.io import imread
from skimage.util import random_noise
from skimage.filters import threshold_otsu, try_all_threshold
from frangiFilter2D import eig2image,FrangiFilter2D

def genImage(img):
    t = img/255.
    t = FrangiFilter2D(t,FrangiScaleRange=np.array([1, 10]), FrangiScaleRatio=1, FrangiBetaOne=0.8, FrangiBetaTwo=15)
    t = t[0] > threshold_otsu(t[0])
    return t * 255
    


original = imread("FrangiFilter/CS.bmp")

sigma = 0.155
noisy = original

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, channel_axis=-1, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

ax[0, 0].imshow(original)
ax[0, 0].axis('off')
ax[0, 0].set_title('Orginal Image')

ax[1,0].imshow(genImage(original))
ax[1,0].axis('off')
ax[1,0].set_title('Orginal Frangi Filter')

tvChambolle = denoise_tv_chambolle(noisy, weight=0.1, channel_axis=-1)
ax[0, 1].imshow(tvChambolle)
ax[0, 1].axis('off')
ax[0, 1].set_title('TV')

ax[1,1].imshow(genImage(tvChambolle))
ax[1,1].axis('off')
ax[1,1].set_title('TV Frangi Filter')

bilaterial = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15)
ax[0, 2].imshow(bilaterial)
ax[0, 2].axis('off')
ax[0, 2].set_title('Bilateral')

ax[1,2].imshow(genImage(bilaterial))
ax[1,2].axis('off')
ax[1,2].set_title('Bilateral Frangi Filter')

wavelet = denoise_wavelet(noisy, rescale_sigma=True)
ax[0, 3].imshow(wavelet)
ax[0, 3].axis('off')
ax[0, 3].set_title('Wavelet denoising')

ax[1,3].imshow(genImage(wavelet))
ax[1,3].axis('off')
ax[1,3].set_title('Wavelet Frangi Filter')


fig.tight_layout()

plt.show()