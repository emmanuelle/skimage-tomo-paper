import matplotlib.pyplot as plt

from skimage import io, segmentation

im = io.imread('ct_wikipedia.jpg', as_grey=True)
pixels = segmentation.felzenszwalb(im, scale=200)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.imshow(segmentation.mark_boundaries(im, pixels))
ax.axis('off')
fig.show()
plt.savefig('super_pixels.png')

