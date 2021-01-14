import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from defs import lowpass_mesh_smoothing

tris = np.genfromtxt('../../data/example_triangles.csv', delimiter=',')
vert = np.genfromtxt('../../data/example_vertices.csv', delimiter=',')

new_vert_5 = lowpass_mesh_smoothing(vert, tris, n_iter=5)
new_vert_10 = lowpass_mesh_smoothing(vert, tris, n_iter=10)
new_vert_25 = lowpass_mesh_smoothing(vert, tris, n_iter=25)
tris_plt = np.genfromtxt('../../data/example_triangles.csv', delimiter=',')-1

fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot_trisurf(vert[:,0], vert[:,1], vert[:,2], triangles=tris_plt)
ax = fig.add_subplot(1,2,2,projection='3d')
ax.plot_trisurf(new_vert_5[:,0], new_vert_5[:,1], new_vert_5[:,2], triangles=tris_plt)

fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot_trisurf(vert[:,0], vert[:,1], vert[:,2], triangles=tris_plt)
ax = fig.add_subplot(1,2,2,projection='3d')
ax.plot_trisurf(new_vert_10[:,0], new_vert_10[:,1], new_vert_10[:,2], triangles=tris_plt)

fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot_trisurf(vert[:,0], vert[:,1], vert[:,2], triangles=tris_plt)
ax = fig.add_subplot(1,2,2,projection='3d')
ax.plot_trisurf(new_vert_25[:,0], new_vert_25[:,1], new_vert_25[:,2], triangles=tris_plt)

plt.show()
