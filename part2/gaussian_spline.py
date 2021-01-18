import numpy as np


class RBFSpline():

    def fit(pts, tpts, lamb):
        ''' Function outputs spline 
        coefficients representing fitted spline'''

        