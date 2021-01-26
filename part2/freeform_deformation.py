import numpy as np
from numpy import linalg
import itertools
from gaussian_spline import RBFSpline

class Image3D():

    def __init__(self, image):

        self.image3d = image


class FreeFormDeformation():

    def __init__(self, r_control, voxdims, n_control = 3):

        self.n_control = n_control
        self.range = np.array(r_control) # [[0,2],[0,2],[0,2]]
        coords_x = np.linspace(self.range[0,0], self.range[0,1], self.n_control)/voxdims[0]
        coords_x = [int(x) for x in coords_x]
        coords_y = np.linspace(self.range[1,0], self.range[1,1], self.n_control)/voxdims[1]
        coords_y = [int(x) for x in coords_y]
        coords_z = np.linspace(self.range[2,0], self.range[2,1], self.n_control)/voxdims[2]
        coords_z = [int(x) for x in coords_z]

        self.coords = list(set(itertools.permutations(np.concatenate((coords_x, coords_y, coords_z),\
            axis=0), self.n_control)))
        self.cont = np.meshgrid(coords_x, coords_y, coords_z)
        #print(self.coords)

    def random_transform_generator(self, s = 0.8):

        if s not in range(0, 2):
            print('s should within the range of [0,1]')

        new_coords = []
        for i in range(len(self.coords)):
            new_coords.append(self.coords[i]+np.random.randint(1,3,3))
        
        return new_coords

    def warp_image(self, image, query):

        rbf = RBFSpline(self.coords, self.random_transform_generator(), self.n_control)

        new_vox = rbf.evaluate(query, cont=self.coords, n=self.n_control)
        warp_image = image
        print(warp_image[9,9,13])
        warp_image[list(query[0])] = new_vox 

        return warp_image

    def random_transform(self, image):

        query = self.random_transform_generator()
        warp_image = self.warp_image(image, query)

        return warp_image


