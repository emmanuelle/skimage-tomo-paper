from skimage import restoration, io
import matplotlib.pyplot as plt

dat = io.imread('phase_separation_insitu.png', as_grey=True)

tv = restoration.denoise_tv_bregman(dat, weight=.8)

l_r, l_c = dat.shape
dat[:, l_c / 2:] = tv[:, l_c / 2:]

plt.imsave('tv_denoising.png', dat[100:350, 245 - 150:245 + 150], cmap='gray', vmin=0.2, vmax=0.8)
