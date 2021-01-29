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

        K = self.kernel_gaussian()
        a = K + lamb*inv(np.identity(n))
        
        coef_x = solve(a, q_x)
        coef_y = solve(a, q_y)
        coef_z = solve(a, q_z)

        return coef_x, coef_y, coef_z

    def evaluate(self, sig=31):
        ''' The function output the transformed query point set. '''
      
        coef_x, coef_y, coef_z = self.fit(pre_trans=self.cont, trans=self.new_cont, n=self.n)
        
        trans_pts = []
        for i in range(self.n):
            trans_pts_x = int(coef_x[i]*np.exp(-(norm(self.query[i,0]-self.cont[i,0]))**2/(2*sig**2)))
            trans_pts_y = int(coef_y[i]*np.exp(-(norm(self.query[i,1]-self.cont[i,1]))**2/(2*sig**2)))
            trans_pts_z = int(coef_z[i]*np.exp(-(norm(self.query[i,2]-self.cont[i,2]))**2/(2*sig**2)))
            trans_pts.append(np.array([trans_pts_x, trans_pts_y, trans_pts_z]))
        
        return trans_pts


    def kernel_gaussian(self, sig=31):
        ''' The function outputs K, the kernel values between the query and the control point sets. '''

        K = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                K[i,j] = np.exp(-(norm(self.cont[i]-self.cont[j]))**2/(2*sig**2))
        print(K)
        return K