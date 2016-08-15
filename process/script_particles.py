import numpy as np
from skimage import segmentation, morphology, measure
from mayavi import mlab


def random_ellipses(l=256, number=30, size_fraction=0.1, seed=None):
    rs = np.random.RandomState(seed)
    im = np.zeros((l, l, l), dtype=np.uint8)
    X, Y, Z = np.ogrid[:l, :l, :l]
    x, y, z = rs.random_integers(0, l, 3 * number).reshape((3, number))
    a, b, c = l * size_fraction / 2 * (rs.rand(3 * number).
                            reshape((3, number))) + l * size_fraction / 2
    for (xx, yy, zz, aa, bb, cc) in zip(x, y, z, a, b, c):
        im += ((X - xx)**2 / aa**2 + (Y - yy)**2 / bb**2 + 
                (Z - zz)**2 / cc**2) < 1
    return segmentation.clear_border(im)


im = random_ellipses(seed=0)
labels = morphology.label(im)
props = measure.regionprops(labels)
vols = np.array([0] + [prop.extent for prop in props])
color_by_extent = vols[labels]

fig = mlab.figure(bgcolor=(0, 0, 0))
src = mlab.pipeline.scalar_field(im)
src.image_data.point_data.add_array(color_by_extent.T.ravel())
# We need to give a name to our new dataset.
src.image_data.point_data.get_array(1).name = 'extent'
src.update()
src2 = mlab.pipeline.set_active_attribute(src, point_scalars='scalar')
contour = mlab.pipeline.contour(src2)
contour2 = mlab.pipeline.set_active_attribute(contour, point_scalars='extent')
mlab.pipeline.surface(contour2, colormap='gnuplot2')

mlab.outline()
mlab.show()
