import numpy as np
from numpy.polynomial import polynomial
from defs import finite_difference_gradient, gradient_descent, second_derivative
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quadratic_polynomial(x, a):

    ''' Computes a multivariate polynomial of degree 2 for three 
    variables. Returns a scalar value.'''     
    
    y = a[1]*x[0]**2 + a[2]*x[1]**2 + a[3]*x[2]**2 + a[4]*x[0]*x[1] + \
        a[5]*x[0]*x[2] + a[6]*x[1]*x[2] + a[7]*x[0] + a[8]*x[1] + a[9]*x[2] + a[0]

    return y

x_init = np.array([-50, -50, 0]).reshape((3,1)) # reasonable guess for y to be min

a = np.random.rand(10,1)*10
y = quadratic_polynomial(x_init, a)

''' Finite difference'''
grad = finite_difference_gradient(quadratic_polynomial, x_init, a)

'''Gradient descent '''
opt_x = gradient_descent(quadratic_polynomial, x_init, a, np.array([1,1,1]), 10000, 0.0000001, finite_difference_gradient)
print(opt_x)
min_y = quadratic_polynomial(opt_x, a)

''' Arithmetic verification of gradient descent result''' 
# Check gradient at points outputted by gradient descent
grad_min = finite_difference_gradient(quadratic_polynomial, opt_x, a)
print(grad_min) # outputs close to 0

# Second derivative check
grad2_min = second_derivative(finite_difference_gradient, quadratic_polynomial, opt_x, a)
#print(grad2_min)

''' Visual verification of gradient descent result '''
fig = plt.figure()
x = y = np.linspace(-100, 100)
x, y = np.meshgrid(x, y)
ax = fig.add_subplot(2,3,1, projection='3d')
ax.plot_surface(x, y, quadratic_polynomial([x,y,-100],a))
ax = fig.add_subplot(2,3,2, projection='3d')
ax.plot_surface(x, y, quadratic_polynomial([x,y,-50],a))
ax = fig.add_subplot(2,3,3, projection='3d')
ax.plot_surface(x, y, quadratic_polynomial([x,y,0],a))
ax = fig.add_subplot(2,3,4, projection='3d')
ax.plot_surface(x, y, quadratic_polynomial([x,y,20],a))
ax = fig.add_subplot(2,3,5, projection='3d')
ax.plot_surface(x, y, quadratic_polynomial([x,y,50],a))
ax = fig.add_subplot(2,3,6, projection='3d')
ax.plot_surface(x, y, quadratic_polynomial([x,y,100],a))

fig = plt.figure()
x1 = np.linspace(opt_x[0]-10, opt_x[0]+10)
x2 = np.linspace(opt_x[1]-10, opt_x[1]+10)
x1, x2 = np.meshgrid(x1, x2)
x3 = [opt_x[2]-10, opt_x[2], opt_x[2]+10]

ax = fig.add_subplot(1, 3, 1, projection='3d') 
ax.plot_surface(x1, x2, quadratic_polynomial([x1,x2,x3[0]],a))

ax = fig.add_subplot(1,3,2, projection='3d')
ax.plot_surface(x1, x2, quadratic_polynomial([x1,x2,x3[1]],a))
ax.scatter(opt_x[0], opt_x[1], min_y, c='r')

ax = fig.add_subplot(1,3,3, projection='3d')
ax.plot_surface(x1, x2, quadratic_polynomial([x1,x2,x3[2]],a))

plt.show()
plt.savefig('task3.png')