import numpy as np
from numpy import linalg
import itertools
from gaussian_spline import RBFSpline

class Image3D():

    def __init__(self, data):

        self.vol = np.array(data['vol'])
        self.voxdims = data['voxdims'][0] # each voxel is 0.8x0.8x3 mm^3


class FreeFormDeformation():

    def __init__(self, r_control, vol, n_control = 8):

        self.n_control = n_control
        self.range = np.array(r_control) # [[55,90],[90,150],[20,20]]
        #coords_x = np.linspace(self.range[0,0], self.range[0,1], self.n_control)/voxdims[0]
        #coords_x = [int(x) for x in coords_x]
        #coords_y = np.linspace(self.range[1,0], self.range[1,1], self.n_control)/voxdims[1]
        #coords_y = [int(x) for x in coords_y]
        #coords_z = np.linspace(self.range[2,0], self.range[2,1], self.n_control)/voxdims[2]
        #coords_z = [int(x) for x in coords_z]
        self.coords = np.array(([55,90,20], [55,150,20], [90,90,20],\
            [90,150,20], [0,0,20], [0,vol.shape[1],20], [vol.shape[0],0,20],\
                [vol.shape[0], vol.shape[1], 20]))
        #self.coords = list(set(itertools.permutations(np.concatenate((coords_x, coords_y, coords_z),\
           # axis=0), self.n_control)))
        #self.cont = np.meshgrid(coords_x, coords_y, coords_z)

    @classmethod
    def from_im(self, vol, n_control=8):
        self.coords = np.array(([55,90,20], [55,150,20], [90,90,20],\
            [90,150,20], [0,0,20], [0,vol.shape[1],20], [vol.shape[0],0,20],\
                [vol.shape[0], vol.shape[1], 20]))

    def random_transform_generator(self, vol, s = 0.8):

        if s > 1:
            print('s should be within the range of [0,1]')

        disp = 10 #voxels
        new_coords = np.array(([65,100,20], [65,140,20], [80,100,20],\
            [80,140,20], [0,0,20], [0,vol.shape[1],20], [vol.shape[0],0,20],\
                [vol.shape[0], vol.shape[1], 20])) # 4 moving points, 4 fixed at the corners
        #new_coords = []
        #for i in range(len(self.coords)):
        #    new_coords.append(self.coords[i]+np.random.randint(1,3,3))
        #print(new_coords)
        return new_coords

    def warp_image(self, image, query):

        rbf = RBFSpline(self.coords, self.random_transform_generator(image), query, self.n_control)

        new_vox = rbf.evaluate(query, cont=self.coords, n=self.n_control)
        print(new_vox)
        warp_image = image
        
        for i in range(self.n_control):
            warp_image[new_vox[i][0], new_vox[i][1], new_vox[i][2]] = warp_image[query[i][0], query[i][1], query[i][2]]

        return warp_image

    def random_transform(self, vol):

        query = np.array(([40,75,20], [40,150,20], [100,75,20],\
            [100,150,20], [0,0,20], [0,vol.shape[1],20], [vol.shape[0],0,20],\
                [vol.shape[0], vol.shape[1], 20]))
        warp_image = self.warp_image(vol, query)

        return warp_image


