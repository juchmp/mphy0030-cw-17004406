import numpy as np
from numpy.linalg import norm, solve, inv

class RBFSpline():

    def __init__(self, cont, new_cont, query, n):

        self.cont = cont # control points
        self.new_cont = new_cont
        self.query = query
        self.n = n

    def fit(self, pre_trans, trans, n, lamb=0.1):
        ''' Function outputs spline 
        coefficients representing fitted spline'''
        
        q_x = trans[:,0]
        q_y = trans[:,1]
        q_z = trans[:,2]

        K = self.kernel_gaussian(self.query, pre_trans)
        a = K + lamb*inv(np.identity(n))
        
        coef_x = solve(a, q_x)
        coef_y = solve(a, q_y)
        coef_z = solve(a, q_z)

        return coef_x, coef_y, coef_z

    def evaluate(self, query, cont, n, sig=5):
        ''' The function output the transformed query point set. '''
      
        coef_x, coef_y, coef_z = self.fit(pre_trans=cont, trans=self.new_cont, n=n)
        
        trans_pts = []
        for i in range(n):
            trans_pts_x = coef_x[i]*np.exp(-(norm(query[i,0]-cont[i,0]))**2/(2*sig**2))
            trans_pts_y = coef_y[i]*np.exp(-(norm(query[i,1]-cont[i,1]))**2/(2*sig**2))
            trans_pts_z = coef_z[i]*np.exp(-(norm(query[i,2]-cont[i,2]))**2/(2*sig**2))
            trans_pts.append(np.array([trans_pts_x, trans_pts_y, trans_pts_z]))
        
        return trans_pts


    def kernel_gaussian(self, query, cont, sig=5):
        ''' The function outputs K, the kernel values between the query and the control point sets. '''

        K = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                K[i,j] = np.exp(-(norm(query[i]-cont[j]))**2/(2*sig**2))

        return K