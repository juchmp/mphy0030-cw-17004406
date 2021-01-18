import numpy as np

class Image3D():

    def __init__(self, image):

        self.image3d = image


class FreeFormDeformation():

    def __init__(self, image3d, n_control = 10, r_control):

        self.image3d = image3d
        self.n_control = n_control
        self.range = r_control

