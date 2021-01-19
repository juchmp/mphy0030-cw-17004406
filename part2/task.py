import numpy as np
from scipy.io import loadmat
#from freeform_deformation import Image3D

data = loadmat('../data/example_image.mat')
vol = np.array(data['vol'])
print(vol)
voxdims = data['voxdims'] # each voxel is 0.8x0.8x3 mm^3
print(np.diag([0.5,0.6,0.7]))
image3d = Image3D(vol) # initiate image3d object

