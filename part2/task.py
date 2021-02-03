import numpy as np
from scipy.io import loadmat
from freeform_deformation import Image3D, FreeFormDeformation
import matplotlib.pyplot as plt

# Image load
data = loadmat('../data/example_image.mat')
image = Image3D(data) # instantiate image3d object

# Instantiating FreeFormDeformation object using range of point distribution
ran_min = [1, 1, 0]
ran_max = [50, 50, 0]
ran = [ran_min, ran_max]
ffd = FreeFormDeformation(ran, image.voxdims)

''' Plotting '''
z = [5, 10, 20, 25, 30]
for i in range(9):
    new_images = ffd.random_transform(image.vol_coords, image.vol)
    fig = plt.figure()
    for j,k in z:
        ax = fig.add_subplot(1,5,j)
        plt.imshow(new_images[:,:,k], cmap='gray')

