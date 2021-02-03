import numpy as np
from numpy import linalg
import itertools
from gaussian_spline import RBFSpline

class Image3D():

    ''' Extracts data from 3D images. '''

    def __init__(self, data):

        self.vol = np.array(data['vol'])
        self.voxdims = data['voxdims'][0] # each voxel is 0.8x0.8x3 mm^3
        vol_x = np.linspace(0, self.vol.shape[0], self.vol.shape[0], dtype=int)
        vol_y = np.linspace(0, self.vol.shape[1], self.vol.shape[1], dtype=int)
        vol_z = np.linspace(0, self.vol.shape[2], self.vol.shape[2], dtype=int)
        self.vol_coords = [vol_x, vol_y, vol_z]


class FreeFormDeformation():

    def __init__(self, ran, voxdims, n_control = 10):

        self.n_control = n_control
        self.range = np.array(ran)
        
        coords_x = np.linspace(self.range[0,0]+50, self.range[1,0]+50, self.n_control)/voxdims[0]
        coords_x = [int(x) for x in coords_x]
        coords_y = np.linspace(self.range[0,1]+50, self.range[1,1]+50, self.n_control)/voxdims[1]
        coords_y = [int(x) for x in coords_y]
        coords_z = np.linspace(self.range[0,2]+20, self.range[1,2]+20, self.n_control)/voxdims[2]
        coords_z = [int(x) for x in coords_z]
        self.cont = [coords_x, coords_y, coords_z] # precomputed control points coordinates
        

    @classmethod
    def from_im(self, vol, n_control=10):
        ''' Optional constructor calling directly on image coordinates.'''

        coords_x = np.linspace(vol[0][50], vol[0][120], n_control)
        coords_y = np.linspace(vol[1][50], vol[1][120], n_control)
        coords_z = np.linspace(vol[2][50], vol[2][120], n_control)
        self.cont = [coords_x, coords_y, coords_z]

    def random_transform_generator(self, s = 0.8):
        ''' Function generates random displaced control points.
        
        Parameters
        ----------
        s : scalar, default = 0.8
            Randomness strength parameter.
            
        Returns
        -------
        new_cont : ndarray, shape(number of control points, 3)
            Randomly transformed control points coordinates. 
            
        '''

        if s > 1:
            print('s should be within the range of [0,1]')

        new_cont = []
        
        for i in range(self.n_control):
            new_cont_x = self.cont[0][i]+np.random.randint(10,50)
            new_cont_y = self.cont[1][i]+np.random.randint(10,50)
            new_cont_z = self.cont[2][i]
            new_cont.append([new_cont_x, new_cont_y, new_cont_z])
        
        return new_cont

    def warp_image(self, query, image, rand_trans):
        ''' Function outputs warped image using Gaussian spline interpolation. 
        
        Parameters
        ----------
        query : ndarray, shape(image.shape, 3)
            Query point set containing image coordinates for each x, y, z direction.
        image : ndarray, shape(image.shape)
            Voxel values containing in the image.
        rand_trans : ndarray, shape(number of control points, 3)
            Randomly transformed control points coordinates.
            
        Returns
        -------
        warp_image : ndarray, shape(image.shape)
            Warped image.
            
        '''

        rbf = RBFSpline(self.cont, rand_trans, query, self.n_control)
        
        new_x, new_y, new_z = rbf.evaluate()
        new_x, new_y, new_z = np.meshgrid(new_x, new_y, new_z)

        warp_image = image[new_x[0], new_y[0], new_z[0]]
            
        return warp_image

    def random_transform(self, query, image):
        ''' Function generates random transformations to warp image. 
        
        Parameters
        ----------
        query : ndarray, shape(image.shape, 3)
            Query point set containing image coordinates for each x, y, z direction.
        image : ndarray, shape(image.shape)
            Voxel values containing in the image.
            
        '''

        rand_trans = self.random_transform_generator()
        warp_image = self.warp_image(query, image, rand_trans)
