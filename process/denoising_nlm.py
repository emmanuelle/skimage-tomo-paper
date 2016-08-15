import numpy as np
from skimage import restoration, io, img_as_float
import matplotlib.pyplot as plt

dat = img_as_float(io.imread('raw_image.jpg'))

nlm = restoration.denoise_nl_means(dat, h=0.1)

l_r, l_c = dat.shape
dat[:, l_c / 2:] = nlm[:, l_c / 2:]

plt.imsave('nlm_denoising.png', dat[30:-30, 30:-30], cmap='gray', vmin=0.1, vmax=0.9)
