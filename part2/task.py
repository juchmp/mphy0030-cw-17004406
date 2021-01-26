import numpy as np
from scipy.io import loadmat
from freeform_deformation import Image3D, FreeFormDeformation
import matplotlib.pyplot as plt

data = loadmat('../data/example_image.mat')
vol = np.array(data['vol'])
#print(vol)
voxdims = data['voxdims'][0] # each voxel is 0.8x0.8x3 mm^3
image3d = Image3D(vol) # initiate image3d object

ran_min = [5*voxdims[0], 5*voxdims[1], 5*voxdims[2]]
ran_max = [11*voxdims[0], 11*voxdims[1], 11*voxdims[2]]
ran = [[ran_min[0], ran_max[0]], [ran_min[1], ran_max[1]], [ran_min[2], ran_max[2]]] # in mm
ffd = FreeFormDeformation(ran, voxdims)

z = [5, 10, 20, 25, 30]
for i in range(9):
    new_images = ffd.random_transform(image=vol)
    fig = plt.figure()
    for j,k in z:
        ax = fig.add_subplot(1,5,k)
        plt.imshow(new_images[:,:,j], cmap='gray')
    plt.show()

