import numpy as np
from numpy.linalg import norm, solve

class RBFSpline():

    def fit(coords, new_coords, lamb, w=0.5, n):
        ''' Function outputs spline 
        coefficients representing fitted spline'''

        lamb = []
        for k in range(n):
            lamb.append(1.5*norm(coords[k,:]-new_coords[k,:])) # as long as bigger than 1.06
        avg_lamb = np.mean(lamb)

        K = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                K[i,j] = np.exp(-norm(coords[i,:]-new_coords[j,:])^2/(2*avg_lamb^2))

        L = np.diag(lamb)
        q_x = new_coords[:,0]
        q_y = new_coords[:,1]
        q_z = new_coords[:,2]

        a = K + w*np.inv(L)
        coef_x = solve(a, q_x)
        coef_y = solve(a, q_y)
        coef_z = solve(a, q_z)

        return coef_x, coef_y, coef_z

    def evaluate()