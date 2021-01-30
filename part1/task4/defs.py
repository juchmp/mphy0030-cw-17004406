import numpy as np

def lowpass_mesh_smoothing(vert, tris, n_iter=10, lamb=0.9, mu=-1.02):
    ''' The function outputs a list of vertices with adjusted
    point coordinates, representing the smoothed surface.'''

    new_points = []

    for i in range(vert.shape[0]):
        pos_idx = np.where(tris == i+1)
        pos = tris[pos_idx[0], :]
        neigh_idx = list(pos[pos != i+1])
        neigh_idx = dict.fromkeys(neigh_idx)
        neigh_idx = [int(x)-1 for x in neigh_idx]
        neigh = vert[neigh_idx,:] # array containing neighbours' x,y,z coord
        
        neigh_nb = neigh.shape[0] # number of neighbours
        cur_pos = vert[i, :] 
        w = 1/neigh_nb
        for j in range(n_iter):

            dist = []
            for k in range(neigh_nb-1):
                dist.append((neigh[k,:] - cur_pos))
            dist_sum = np.sum(dist, axis=0)
            new_pos = cur_pos + lamb*(w*dist_sum)

            dist = []
            for k in range(neigh_nb-1):
                dist.append((neigh[k,:] - new_pos))
            dist_sum = np.sum(dist,axis=0)
            final_pos = new_pos + mu*lamb*(w*dist_sum)
            cur_pos = final_pos

        new_points.append(cur_pos)

    return np.array(new_points)

        
    

    
       