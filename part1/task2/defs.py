import numpy as np
from numpy.linalg import det, multi_dot, inv
from numpy import pi, exp, matmul, sqrt, var, log, abs
from scipy.stats import norm

def bi_gaussian_pdf(x, mu, sig):

    N = 2*pi*sqrt(det(sig))
    mat_mul = np.einsum('...k,kl,...l->...', x-mu, inv(sig), x-mu)
    pdf = exp(-1/2 * mat_mul) / N
    #for i in range(x.shape[0]):
     #   p = 1 / (2*pi*sqrt(det(sig))) \
      #      * exp(-1/2*multi_dot([np.atleast_2d(x[i,:]-mu), inv(sig), np.atleast_2d(x[i,:]-mu).T]).astype('float128'))
       # pdf.append(p)
    #pdf = np.array(pdf)

    return pdf

def solve_eq(x, perc):

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
    
    width = abs(roots[0]-roots[1])
    height = abs(roots[2]-roots[3])
    xcord = (width)/2 + roots[1]
    ycord = (height)/2 + roots[3]

    return xcord, ycord, width