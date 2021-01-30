import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from defs import lowpass_mesh_smoothing

''' Loading data '''
tris = np.genfromtxt('../../data/example_triangles.csv', delimiter=',')
vert = np.genfromtxt('../../data/example_vertices.csv', delimiter=',')
tris_plt = np.genfromtxt('../../data/example_triangles.csv', delimiter=',')-1

''' Result of mesh smoothing '''
new_vert_5 = lowpass_mesh_smoothing(vert, tris, n_iter=5)
new_vert_10 = lowpass_mesh_smoothing(vert, tris, n_iter=10)
new_vert_25 = lowpass_mesh_smoothing(vert, tris, n_iter=25)

''' Plotting '''

fig = plt.figure()

ax = fig.add_subplot(2,2,1,projection='3d')
ax.plot_trisurf(vert[:,0], vert[:,1], vert[:,2], triangles=tris_plt)
ax.title.set_text('Original surface')
ax = fig.add_subplot(2,2,2,projection='3d')
ax.plot_trisurf(new_vert_5[:,0], new_vert_5[:,1], new_vert_5[:,2], triangles=tris_plt)
ax.title.set_text('After 5 iterations')
ax = fig.add_subplot(2,2,3,projection='3d')
ax.plot_trisurf(new_vert_10[:,0], new_vert_10[:,1], new_vert_10[:,2], triangles=tris_plt)
ax.title.set_text('After 10 iterations')
ax = fig.add_subplot(2,2,4,projection='3d')
ax.plot_trisurf(new_vert_25[:,0], new_vert_25[:,1], new_vert_25[:,2], triangles=tris_plt)
ax.title.set_text('After 25 iterations')
fig.tight_layout()

#plt.show()
plt.savefig('task4.png')