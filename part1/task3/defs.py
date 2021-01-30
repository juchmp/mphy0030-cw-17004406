import numpy as np

def gradient_descent(f, var, coeffs, step, n_iter, tol, grad_func):

    ''' The function outputs the optimal function variable value  
 that minimises the multivariate function (output) value. '''
    count = 0
    rate = 0.01
   
    while min(step) > tol and count < n_iter:
        pre_vars = var
        if (grad_func(f, pre_vars, coeffs) >= 0).all():
            var = var - rate*grad_func(f, pre_vars, coeffs)
        else:
            var = var + rate*grad_func(f, pre_vars, coeffs)
        step = abs(var - pre_vars)
        count = count + 1
    if count > n_iter:
        print('Maximum number of iterations reached')
    else: 
        print('Step size now too small compared to tolerance value')
    return var

def finite_difference_gradient(f, var, coeffs):

    '''The function outputs all the estimated partial  
    derivatives for the input vector variable.'''

    grad = []
    h = 1e-5
    for i, _ in enumerate(var):
        d = []
        for j, vars_j in enumerate(var):
            if i==j:
                d.append(vars_j + h)
            else:
                d.append(vars_j) 
        diff = (f(d, coeffs) - f(var, coeffs)) / h
        grad.append(diff)

    return np.array(grad)

def second_derivative(gradf, f, var, coeffs):
    ''' Function outputting the first derivative of the given quadratic polynomial '''

    h = 1e-5
    grad_at_opt = gradf(f, var, coeffs)
    grad_at_h = gradf(f, var+h, coeffs) 
    diff = (grad_at_h - grad_at_opt)/h

    return diff
