from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from defs import simple_image_write, simple_image_read

mat_data = loadmat('../../data/example_image.mat')
vol = np.array(mat_data['vol'], dtype='int16')
#print(vol[:50])
voxdims = mat_data['voxdims'].astype('float32')

file_to_read = simple_image_write(vol)

vol_bin = simple_image_read(file_to_read) # currently a 1d array but want 3d array
vol_bin = np.reshape(vol_bin, vol.shape)

## Plots

# First choice of z-coordinate
slice1 = vol_bin[:,:,20]
plt.imshow(slice1,cmap='gray')
plt.show()
