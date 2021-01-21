import numpy as np
from scipy.io import loadmat
from freeform_deformation import Image3D, FreeFormDeformation
import matplotlib.pyplot as plt

data = loadmat('../data/example_image.mat')
vol = np.array(data['vol'])
print(vol)
voxdims = data['voxdims'] # each voxel is 0.8x0.8x3 mm^3

image3d = Image3D(vol) # initiate image3d object


z = [5, 10, 20, 25, 30]
for i in range(9):
    new_images = FreeFormDeformation.random_transform(image3d)
    fig = plt.figure()
    for j,k in z:
        ax = fig.add_subplot(1,5,k)
        plt.imshow(new_images[:,:,j], cmap='gray')
    plt.show()

