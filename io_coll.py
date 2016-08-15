>>> from skimage import io

>>> image_collection = io.imread_collection('*.tif')
>>> image_3d = image_collection.concatenate()
>>> print(image_3d.shape)
(800, 1024, 1024)
