import numpy as np
from scipy.io import loadmat
from freeform_deformation import Image3D

data = loadmat('../data/example_image.mat')
vol = np.array(data['vol'])
voxdims = data['voxdims'] # each voxel is 0.8x0.8x3 mm^3

image3d = Image3D(vol) # initiate image3d object
