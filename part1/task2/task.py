import numpy as np
from numpy import random as rd
from defs import gaussian_pdf, solve_eq, circle_cords
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D, art3d

# Randomly generating 10,000 samples of vector x
x = rd.rand(10000, 3) 

# Means and covariance matrices
mu = np.mean(x, axis=0)
sig = np.cov(x, rowvar=False)

# Probability densities and percentiles
pdf = gaussian_pdf(x, mu, sig)
perc10, perc50, perc90 = np.percentile(pdf, [10,50,90])

# First step in plotting percentiles 
# Finding the coordinates of the probability densities equal to each percentile.
roots10 = []
roots50 = []
roots90 = []
for i in range(x.shape[1]):
    roots10.append(solve_eq(x[:,i], perc10))
    roots50.append(solve_eq(x[:,i], perc50))
    roots90.append(solve_eq(x[:,i], perc90))

''' Plotting of the surfaces representing the respective percentiles '''

x1 = np.sort(x[:,0], axis=None)
x2 = np.sort(x[:,1], axis=None)
x1, x2 = np.meshgrid(x1, x2)
pdf[::-1].sort()
pdf = np.meshgrid(pdf, pdf)

fig = plt.figure()
ax = fig.gca(projection = '3d')

# Surface of cdf to visualise the probability densities
plot = ax.plot_surface(x1, x2, pdf[0], linewidth=1, label='CDF equivalent of computed pdf')
plot._edgecolors2d = plot._edgecolors3d
plot._facecolors2d = plot._facecolors3d

# Circle plots of respective percentiles
xcord90, ycord90, width90 = circle_cords(roots90)
c90 = Circle((xcord90,ycord90), width90/2, alpha=0.8, color='g', label='90th percentile')
xcord50, ycord50, width50 = circle_cords(roots50)
c50 = Circle((xcord50,ycord50), width50/2, alpha=0.8, color='r', label='50th percentile')
xcord10, ycord10, width10 = circle_cords(roots10)
c10 = Circle((xcord10,ycord10), width10/2, alpha=0.8, color='y', label='10th percentile')

ax.add_patch(c10)
ax.add_patch(c50)
ax.add_patch(c90)
art3d.patch_2d_to_3d(c10, z=perc10, zdir='z')
art3d.patch_2d_to_3d(c50, z=perc50, zdir='z')
art3d.patch_2d_to_3d(c90, z=perc90, zdir='z')

ax.legend()

#plt.show()
plt.savefig('task2.png')
