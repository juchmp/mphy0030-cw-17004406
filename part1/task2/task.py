import numpy as np
from numpy import random as rd
from defs import bi_gaussian_pdf, solve_eq, circle_cords
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D, art3d

x = rd.rand(2, 10000) # randomly generating 10,000 samples of vector x
x1 = np.sort(x[0,:], axis=None)
x2 = np.sort(x[1,:], axis=None)
x1, x2 = np.meshgrid(x1, x2)

mu = np.mean(x, axis=1)
sig = np.cov(x)

pos = np.empty(x1.shape + (2,), dtype='float32') # creates empty array of 10000 by 10000 by 2
pos[:,:,0] = x1 # set first 10000 by 10000 grid to x1 values and the next to x2 values
pos[:,:,1] = x2

pdf = bi_gaussian_pdf(pos, mu, sig)

perc10, perc50, perc90 = np.percentile(pdf, [10,50,90])


r1x1_10, r2x1_10 = solve_eq(x1, perc10)
r1x2_10, r2x2_10 = solve_eq(x2, perc10)
roots10 = np.array([r1x1_10, r2x1_10, r1x2_10, r2x2_10])

r1x1_50, r2x1_50 = solve_eq(x1, perc50)
r1x2_50, r2x2_50 = solve_eq(x2, perc50)
roots50 = np.array([r1x1_50, r2x1_50, r1x2_50, r2x2_50])

r1x1_90, r2x1_90 = solve_eq(x1, perc90)
r1x2_90, r2x2_90 = solve_eq(x2, perc90)
roots90 = np.array([r1x1_90, r2x1_90, r1x2_90, r2x2_90])

# Plots

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(x1, x2, pdf, linewidth=1)

xcord10, ycord10, width10 = circle_cords(roots10)
c10 = Circle((xcord10,ycord10), width10/2, fill=0, color='r')
xcord50, ycord50, width50 = circle_cords(roots50)
c50 = Circle((xcord50,ycord50), width50/2, fill=0, color='y')
xcord90, ycord90, width90 = circle_cords(roots90)
c90 = Circle((xcord90,ycord90), width90/2, fill=0, color='g')

ax.add_patch(c10)
ax.add_patch(c50)
ax.add_patch(c90)
art3d.patch_2d_to_3d(c10, z=perc10, zdir='z')
art3d.patch_2d_to_3d(c50, z=perc50, zdir='z')
art3d.patch_2d_to_3d(c90, z=perc90, zdir='z')

#ax.plot_surface(x1[pdf==perc90], x2[pdf==perc90], cont90)
#ax.scatter(x1, x2, cont90, zdir='z')
#ax.set_zlim(0,2)
#ax.set_zticks(np.linspace(0,1,5))
#ax.view_init(27, -21)
plt.show()

