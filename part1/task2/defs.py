import numpy as np
from numpy.linalg import det, multi_dot, inv
from numpy import pi, exp, matmul, sqrt, var, log, abs
from scipy.stats import norm

def gaussian_pdf(x, mu, sig):

    ''' The function outputs the probability density for each 
    set of vector x.
    
    Parameters
    ----------
    x : array-like, shape(number of samples, number of variables)
        Samples of vector x.
    mu : array-like, shape(number of variables,)
        A vector of the samples' means for each variable.
    sig : array-like, shape(number of variables, number of variables)
        Covariance matrix.
        
    Returns
    -------
    pdf : array-like, shape(number of samples,)
        Probability densities for each sample of vector x.'''

    N = ((2*pi)**(3/2))*sqrt(det(sig))
    a = np.matmul(x-mu, inv(sig))
    mat_mul = np.einsum('jk, kj -> j', a, (x-mu).T)
    pdf = exp(-1/2 * mat_mul) / N

    return pdf

def solve_eq(x, perc):

    ''' The function outputs the roots solution to a quadratic equation
    which represents the univariate probability density function when 
    equal to each percentile. 
    Rearranging we get: ax^2 + bx + c = 0 with a, b, c described below.
    
    Parameters
    ----------
    x : array-like, shape(number of samples,)
        Function variables.
    perc : scalar
        Percentile to be evaluated.
    
    Returns 
    -------
    r1, r2 : scalars 
        Roots solving quadratic equation.
    
    '''
    a = 1
    b = -2*np.mean(x)
    c = np.mean(x)**2 + 2*var(x)*log(perc*sqrt(var(x)*2*pi)) 
    d = abs(b**2-4*a*c)

    if d > 0:
        r1 = abs((-b+sqrt(d))/(2*a))
        r2 = abs((-b-sqrt(d))/(2*a))

    if d == 0:
        r1 = r2 = abs((-b+sqrt(d))/(2*a))

    return r1, r2

def circle_cords(roots):
    
    ''' Function outputs the parameters needed to draw circles for each percentile.
    
    Parameters
    ----------
    roots : array-like, shape(number of variables, 2)
        Vector containing the roots in each direction.
    
    Returns
    -------
    xcord : scalar
        X coordinate of the circle's center.
    ycord : scalar
        Y coordinate of the circle's center.
    width : scalar
        Radius of the circle.
    
    '''
    width = abs(roots[0][0]-roots[0][1])
    height = abs(roots[1][0]-roots[1][1])
    xcord = (width)/2 + roots[0][1]
    ycord = (height)/2 + roots[1][1]

    return xcord, ycord, width