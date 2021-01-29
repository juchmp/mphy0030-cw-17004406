import numpy as np
from scipy.io import loadmat
from freeform_deformation import Image3D, FreeFormDeformation
import matplotlib.pyplot as plt

data = loadmat('../data/example_image.mat')
image = Image3D(data) # instantiate image3d object

ran_min = [55, 90, 20]
ran_max = [90, 150, 20]
ran = [[ran_min[0], ran_max[0]], [ran_min[1], ran_max[1]], [ran_min[2], ran_max[2]]] # in mm
ffd = FreeFormDeformation(ran, image.vol)

new_image = ffd.random_transform(vol=image.vol)
fig = plt.figure()
plt.imshow(new_image[:,:,20], cmap='gray')
plt.show()
#z = [5, 10, 20, 25, 30]
#for i in range(9):
 #   new_images = ffd.random_transform(image=image.vol)
  #  fig = plt.figure()
   # for j,k in z:
    #    ax = fig.add_subplot(1,5,k)
     #   plt.imshow(new_images[:,:,j], cmap='gray')

