import numpy as np
from numpy.polynomial import polynomial
from defs import quadratic_polynomial, finite_difference_gradient, gradient_descent, second_derivative
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Initialising parameters
x_init = np.array([100, 100, 100]).reshape((3,1))
a = np.random.rand(10,1)*10
y_init = quadratic_polynomial(x_init, a)

''' Finite difference'''
grad = finite_difference_gradient(quadratic_polynomial, x_init, a)

'''Gradient descent '''
n_iter = 100000
min_x = gradient_descent(quadratic_polynomial, x_init, a, np.array([1,1,1]), n_iter, 0.00001, finite_difference_gradient)
min_y = quadratic_polynomial(min_x, a)

''' Arithmetic verification of gradient descent result''' 
# Check gradient at points outputted by gradient descent
grad_min = finite_difference_gradient(quadratic_polynomial, min_x, a)
print('\nFirst derivative at gradient descent minimum computed:\n', grad_min) # outputs close to 0

# Second derivative sign check
grad2_min = second_derivative(finite_difference_gradient, quadratic_polynomial, min_x, a)
if grad2_min.all() > 0:
    print('The gradient descent result is a local minimum due to positive second derivative \n')
if grad2_min.all() == 0:
    print('The gradient descent result is at an inflexion point due to second derivative = 0 \n Please run again to try a different polynomial')

''' Visual verification of gradient descent result '''
fig = plt.figure()

# Pre gradient descent
x1 = np.linspace(x_init[0]-100, x_init[0]+100)
x2 = np.linspace(x_init[1]-100, x_init[1]+100)
x1, x2 = np.meshgrid(x1, x2)

ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(x1, x2, quadratic_polynomial([x1,x2,x_init[2]],a), alpha=0.9)
ax.scatter(x_init[0], x_init[1], y_init, c='r', marker='x')
ax.set_title('Starting point of gradient descent \n (x3 = %d)' % x_init[2], fontsize=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.tick_params(labelsize=8)

# Post gradient descent
min_x1 = np.linspace(min_x[0]-100, min_x[0]+100)
min_x2 = np.linspace(min_x[1]-100, min_x[1]+100)
min_x1, min_x2 = np.meshgrid(min_x1, min_x2)

ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(min_x1, min_x2, quadratic_polynomial([min_x1, min_x2, min_x[2]],a), alpha=0.9)
ax.scatter(min_x[0], min_x[1], min_y, c='r', marker='x')
ax.set_title('Gradient descent result (x3 = %d) \n Max number of iterations set = ' % min_x[2] + str(n_iter), fontsize=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.tick_params(labelsize=8)

#plt.show()
plt.savefig('task3.png')