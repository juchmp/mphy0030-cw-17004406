import numpy as np
import itertools
from gaussian_spline import RBFSpline

class Image3D():

    def __init__(self, image):

        self.image3d = image


class FreeFormDeformation():

    def __init__(self, n_control = 3, r_control, voxdims):

        self.n_control = n_control
        self.range = r_control/voxdims # [[0,2],[0,2],[0,2]]

        coords_x = np.linspace(self.range[0,0], self.range[0,1], self.n_control)
        coords_y = np.linspace(self.range[1,0], self.range[1,1], self.n_control)
        coords_z = np.linspace(self.range[2,0], self.range[2,1], self.n_control)
        self.coords = np.array(itertools.permutations(np.concatenate((coords_x, coords_y, coords_z),\
            axis=None), self.n_control))

    def random_transform_generator(self, s = 0.8):

        if s not in range(0, 2):
            print('s should within the range of [0,1]')

        new_coords = []
        for i in range(self.coords.shape[0]):
            new_coords.append(self.coords[i,:]+np.random.randint(1,3,3))
        
        return new_coords

    def warp_image(self, image, query):

        image_obj = Image3D(image)
        rbf = RBFSpline(self.coords, self.random_transform_generator(), self.n_control)

        new_vox = rbf.evaluate(query)
        warp_image = image_obj
        warp_image[query] = new_vox 

        return warp_image

    def random_transform(self, image):

        query = self.random_transform_generator()
        warp_image = self.warp_image(image, query)

        return warp_image


