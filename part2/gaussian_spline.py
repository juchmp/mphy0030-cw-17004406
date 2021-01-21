import numpy as np
from numpy.linalg import norm, solve

class RBFSpline():

    def __init__(self, cont, new_cont, n):

        self.cont = cont # control points
        self.new_cont = new_cont
        self.n = n

    def fit(self, pre_trans=self.cont, trans=self.new_cont, w=0.5, n=self.n):
        ''' Function outputs spline 
        coefficients representing fitted spline'''
        
        lamb = []
        for k in range(n):
            lamb.append(1.5*norm(pre_trans[k,:]-trans[k,:])) # as long as bigger than 1.06

        L = np.diag(lamb)
        q_x = trans[:,0]
        q_y = trans[:,1]
        q_z = trans[:,2]

        K = self.kernel_gaussian(pre_trans, trans)
        a = K + w*np.inv(L)

        coef_x = solve(a, q_x)
        coef_y = solve(a, q_y)
        coef_z = solve(a, q_z)

        return coef_x, coef_y, coef_z

    def evaluate(self, test, cont=self.cont, n=self.n):
        ''' The function output the transformed query point set. '''

        coef_x, coef_y, coef_z = self.fit()

        lamb = []
        for k in range(n):
            lamb.append(1.5*norm(cont[k,:]-self.new_cont[k,:])) # as long as bigger than 1.06
        avg_lamb = np.mean(lamb)

        trans_pts_x = []
        trans_pts_y = []
        trans_pts_z = []

        for i in range(n):
            trans_pts_x.append(coeff_x[i]*np.exp(-norm(test[i,0]-cont[i,0])^2/(2*avg_lamb^2)))
            trans_pts_y.append(coeff_y[i]*np.exp(-norm(test[i,1]-cont[i,1])^2/(2*avg_lamb^2)))
            trans_pts_z.append(coeff_z[i]*np.exp(-norm(test[i,2]-cont[i,2])^2/(2*avg_lamb^2))))

        return trans_pts_x, trans_pts_y, trans_pts_z


    def kernel_gaussian(self, test, cont, sig=1):
        ''' The function outputs K, the kernel values between the query and the control point sets. '''

        K = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                K[i,j] = np.exp(-norm(test[i,:]-cont[j,:])^2/(2*sig^2))

        return K