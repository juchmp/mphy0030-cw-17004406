from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from defs import simple_image_write, simple_image_read


# Image loading
mat_data = loadmat('../../data/example_image.mat')
vol = np.array(mat_data['vol'], dtype='int16')
voxdims = mat_data['voxdims'].astype('float32')

# Image writing
file_to_read = simple_image_write(vol)

# Image reading
vol_bin = simple_image_read(file_to_read) 
vol_bin = np.reshape(vol_bin, vol.shape)

''' Plotting '''

z = [10, 15, 20]
fig, ax = plt.subplots(1, 3)
for i, j in enumerate(z):
    ax[i%3].imshow(vol_bin[:,:,j],cmap='gray')
    ax[i%3].set_title('Slice z = %d' %j)
fig.tight_layout(pad=2.0)
#plt.show()
plt.savefig('task1.png')