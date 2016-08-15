import numpy as np
import matplotlib.pyplot as plt

from skimage import io, segmentation

im = io.imread('ct_wikipedia.jpg', as_grey=True)
#pixels = segmentation.slic(im, n_segments=600, compactness=2.e-4)
pixels = segmentation.felzenszwalb(im, scale=200)

plt.figure(figsize=(5, 5))
plt.imshow(segmentation.mark_boundaries(im, pixels))
plt.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('super_pixels.png')

