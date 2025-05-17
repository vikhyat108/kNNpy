####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
import scipy
import scipy.spatial
import gc
import scipy.spatial
import utils as il
import time
from HelperFunctions.py import create_query_3D
####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

####################################################################################################
def get_knn_dists(pos, query_pos, n_kNN, boxsize):
    '''
    Function that builds a kD-tree and finds the k nearest neighbors for each query point.
    Inputs:
        pos: shape (N,3) array of positions, where N is the number of data points 
        a numpy array of data points

        query_pos: shape (M,3) array of positions, where M is the number of query points
        a numpy array of query points

        kNN: number of nearest neighbors
        
        boxsize: float
        size of the box
    Outputs:
        dis: shape (M,kNN) array of distances, where M is the number of query points 
        a numpy array of distances to the k nearest neighbors. 
    '''
    xtree    = scipy.spatial.cKDTree(pos, boxsize=boxsize)
    dis, idx = xtree.query(query_pos, k=n_kNN, workers = -1)
    del(idx)
    gc.collect()
    return dis
####################################################################################################
#--------------------------------------  Main Code Block  -----------------------------------------
def kNN_3D_TracerAuto3D(pos, n_kNN, query_type, query_grid, boxsize):
    '''
    Gives the kNN CDFs for a given set of data points in 3D space.
    Inputs:
    query_type : {'grid', 'random'}, str
        the type of query points to be generated; should be 'grid' for query points defined on a uniform grid and 'random' for query points drawn from a uniform random distribution.
    query_grid : int
        the 1D size of the query points array; the total number of query points generated will be ``query_grid**3``.
    pos: numpy array of shape (N,3)
        the positions of the data points, where N is the number of data points.
    n_kNN: int
        the number of nearest neighbors to be considered for each query point.
    boxsize: float
        the size of the box in which the data points are located.
    Outputs:
        cutoff_intp_cdf: interpolated cumulative distribution function in (0.05, 0.95)
        intp_cdf: interpolated cumulative distribution function
    '''

    if query_grid**3>=100*pos.shape[0]:
        
        query_pos=create_query_3D(query_type, query_grid, boxsize)
        r=np.linspace(0,boxsize,query_grid**3)  #length scale
        dist=get_knn_dists(pos,query_pos,n_kNN,boxsize)
        dist=np.sort(dist)

        cdf=[]
    for k in range(len(r)):
    # Count how many of the k-NN distances are less than the current radius r[k]
        count = np.sum(dist <= r[k])
        prob = count / query_grid**3  # Probability is the count divided by the total number of query points
        cdf.append(prob)

        return cdf
    else:
        print("Warning: The number of query points is less than 100 times the number of data points. This may lead to inaccurate results.")


