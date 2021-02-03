import numpy as np
from numpy.linalg import norm, solve, inv

class RBFSpline():

    ''' Fits and evaluates a Gaussian RBF spline interpolation function. '''

    def __init__(self, cont, new_cont, query, n):

        self.cont = np.array(cont)
        self.new_cont = new_cont
        self.query = query
        self.n = n

    def fit(self, lamb=0.01):
        ''' Function outputs spline coefficients representing fitted spline. 
        
        Parameters
        ----------
        lamb : scalar, default = 0.01
            Weighting parameter representing the localization errors.
            
        Returns
        -------
        coef_x, coef_y, coef_z : array-like, shape(number of control points,)
            Spline coefficients for each x, y, z directions.
            
        '''

        # Displaced control points
        trans=np.array(self.new_cont)
        q_x = np.array(trans[:,0])
        q_y = trans[:,1]
        q_z = trans[:,2]

        # Gaussian kernel for each x, y, z direction
        Kx, Ky, Kz = self.kernel_gaussian(query=self.cont)
        ax = Kx + lamb*inv(np.identity(self.n))
        ay = Ky + lamb*inv(np.identity(self.n))
        az = Kz + lamb*inv(np.identity(self.n))
        
        # Solving linear system of matrices to obtain coefficients - Report Q2!
        coef_x = solve(ax, q_x)
        coef_y = solve(ay, q_y)
        coef_z = solve(az, q_z)
        
        return coef_x, coef_y, coef_z

    def evaluate(self, sig=1):
        ''' The function outputs the transformed query point set. 
        
        Parameters
        ----------
        sig : scalar, default=1
            Gaussian kernel parameter.
            
        Returns
        -------
        new_x, new_y, new_z : matrices, shapes (image.shape[0],) (image.shape[1],) (image.shape[2],)
            Transformed coordinates of the query point set.
            
        '''
        
        # Spline coefficients fitted
        coef_x, coef_y, coef_z = self.fit()
        
        trans = []
        trans_x = []
        trans_y = []
        trans_z = []
        
        for k in range(self.n):
            trans_x.append(coef_x[k]*np.exp(-(abs(self.query[0]-self.cont[0][k]))**2/(2*sig**2)))
            trans_y.append(coef_y[k]*np.exp(-(abs(self.query[1]-self.cont[1][k]))**2/(2*sig**2)))
            trans_z.append(coef_z[k]*np.exp(-(abs(self.query[2]-self.cont[2][k]))**2/(2*sig**2)))
        # final displacement in each x,y,z direction following interpolation added to original coordinates
        new_x = self.query[0] + np.sum(trans_x, axis=0)
        new_x = [int(x) for x in new_x]
        new_y = self.query[1] + np.sum(trans_y, axis=0)
        new_y = [int(x) for x in new_y]
        new_z = self.query[2] + np.sum(trans_z, axis=0)
        new_z = [int(x) for x in new_z]
        
        return new_x, new_y, new_z


    def kernel_gaussian(self, query, sig=1):
        ''' The function outputs K, the kernel values between the query and the control point sets. 
        
        Parameters
        ----------
        query : array-like, shape(number of control points,)
            Query point set.
        sig : scalar, default = 1
            Gaussian kernel parameter.

        Returns
        -------
        Kx, Ky, Kz : matrices, shape(number of control points, number of control points)
            Gaussian kernel matrices for each x, y, z direction.
        
        '''

        Kx = np.empty((self.n, self.n))
        Ky = np.empty((self.n, self.n))
        Kz = np.empty((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                Kx[i,j] = np.exp(-(abs(query[0][i]-self.cont[0][j]))**2/(2*sig**2))
                Ky[i,j] = np.exp(-(abs(query[1][i]-self.cont[1][j]))**2/(2*sig**2))
                Kz[i,j] = np.exp(-(abs(query[2][i]-self.cont[2][j]))**2/(2*sig**2))

        return Kx, Ky, Kz