import numpy as np

def quadratic_polynomial(x, a):

    ''' Computes a multivariate polynomial of degree 2 for three 
    variables. Returns a scalar value.
    
    Parameters
    ----------
    x : array-like, shape(number of function variables,)
        Vector of polynomial variables.
    a : array-like, shape(number of parameters,)
        Vector of polynomial parameters.
        
    Returns
    -------
    y : scalar
        Polynomial output.
    '''     
    
    y = a[1]*x[0]**2 + a[2]*x[1]**2 + a[3]*x[2]**2 + a[4]*x[0]*x[1] + \
        a[5]*x[0]*x[2] + a[6]*x[1]*x[2] + a[7]*x[0] + a[8]*x[1] + a[9]*x[2] + a[0]

    return y

def gradient_descent(f, var, coeffs, step, n_iter, tol, grad_func):

    ''' The function outputs the optimal function variable value  
    that minimises the multivariate function (output) value. 
    
    Parameters
    ----------
    f : function
        Function to be minimised.
    var : array-like, shape(number of function variables,)
        Vector of polynomial variables.
    coeffs : array-like, shape(number of parameters,)
        Vector of polynomial parameters.
    step : array-like
        Step size for the first iteration of gradient descent.
    n_iter : scalar
        Maximum number of iterations.
    tol : scalar
        Minimum tolerated step size.
    grad_func : function
        Function computing the polynomial's partial derivatives at given point. 
        
    Returns
    -------
    var : array-like, shape(number of variables,)
        The set of variables that minimises the given function. 
    '''

    count = 0
    rate = 0.01

    while max(step) > tol and count < n_iter:
        new_var = np.empty(var.shape)
        new_var = var - rate*grad_func(f, var, coeffs)
        step = abs(var - new_var)
        var = new_var
        count = count + 1
    if count >= n_iter:
        print('Gradient descent stopped due to maximum number of iterations reached')
    else: 
        print('Gradient descent stopped due to step size smaller than tolerance')
    return var

def finite_difference_gradient(f, var, coeffs):

    '''The function outputs all the estimated partial  
    derivatives for the input vector variable.
    
    Parameters
    ----------
    f : function
        Function which partial derivatives will be computed from.
    var : array-like, shape(number of variables,)
        Vector of polynomial variables.
    coeffs : array-like, shape(number of parameters,)
        Vector of polynomial parameters.
        
    Returns
    -------
    grad : array-like, shape(number of variables,)
        Vector of partial derivatives with input variables var.
    '''

    grad = []
    h = 1e-6
    for i, _ in enumerate(var):
        d1 = []
        d2 = []
        for j, vars_j in enumerate(var):
            if i==j:
                d1.append(vars_j + h)
                d2.append(vars_j - h)
            else:
                d1.append(vars_j) 
                d2.append(vars_j)
        diff = (f(d1, coeffs) - f(d2, coeffs)) / (2*h)
        grad.append(diff)

    return np.array(grad)

def second_derivative(gradf, f, var, coeffs):
    ''' The function outputs the second derivative of the given 
    quadratic polynomial using finite difference approximation. 
    
    Parameters
    ----------
    gradf : function
        Function that outputs the first derivative.
    f : function
        Function for which second derivative is computed.
    var : array-like, shape(number of variables,)
        Vector of polynomial variables.
    coeffs : array-like, shape(number of parameters,)
        Vector of polynomial parameters.
    
    Returns
    -------
    diff : array-like, shape(number of variables,)
        Finite difference approximation of second derivative.
    '''

    h = 1e-5
    grad_at_opt = gradf(f, var, coeffs)
    grad_at_h = gradf(f, var+h, coeffs) 
    diff = (grad_at_h - grad_at_opt)/h

    return diff
