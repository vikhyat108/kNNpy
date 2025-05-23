####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
import scipy
import scipy.spatial
import time
import sys
import os
#Importing the required helper function
from kNN_ASMR.HelperFunctions import calc_kNN_CDF
####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

def TracerAuto3D(boxsize, kList, BinsRad, QueryPos, TracerPos, ReturnNNdist=False,Verbose=False):
    
    r'''
    Computes the $k$NN-CDFs in 3D coordinates (Banerjee & Abel (2021)[^1]) of the provided discrete tracer set (`TracerPos`), 
    evaluated at the provided radial distance scales `BinsRad`, for all $k$ in `kList`. Each $k$NN-CDF measures the probability
    $P_{\geq k}(r)$ of finding at least $k$ tracers in a randomly placed sphere of radius $r$. The $k$NN-CDFs quantify the spatial 
    clustering of the tracers.
    		
    Parameters
    ----------
    kList : list of ints
        the list of nearest neighbours to calculate the distances to. For example, if ``kList = [1, 2, 4]``, the first, second and 
        fourth-nearest neighbour distributions will be computed.
    BinsRad : list of numpy float array
        list of radial distance arrays (in Mpc/h) for each nearest neighbour. The $i^{th}$ element of the 
        list should contain a numpy array of the desired distances for the nearest neighbour specified by the $i^{th}$ element of `kList`.
    QueryPos : numpy float array of shape ``(n_query, 3)``
        array of 3D locations for the query points. The 3D locations must be on a grid. The format is (x,y,z) Cartesian coordinates. 
        Please ensure $0<x,y,z<boxsize$.
    TracerPos : numpy float array of shape ``(n_tracer, 3)``
        array of 3D locations for the discrete tracers. The 3D locations must be on a grid. The format is (x,y,z) Cartesian coordinates. 
        Please ensure $0<x,y,z<boxsize$.
    ReturnNNdist : bool, optional
        if set to ``True``, the sorted arrays of NN distances will be returned along with the $k$NN-CDFs, by default ``False``.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    kNN_results: tuple of lists or list of numpy float arrays
        results of the kNN computation. If `ReturnNNdist` is ``True``, returns the tuple ``(p_gtr_k_list, vol)`` where `p_gtr_k_list` 
        is the list of auto kNN-CDFs, and `vol` is the list of NN distances. If `ReturnNNdist` is ``False``, returns `p_gtr_k_list` only
        
    Raises
    ------
    ValueError
        if the given query points are not on a three-dimensional grid.
    ValueError
        if x,y, or z coordinate of any of the query points is not in ``(0, boxsize)``.
    ValueError
        if x,y, or z coordinate of any of the tracer points is not in ``(0, boxsize)``..
    ValueError
        if the given tracer points are not on a three-dimensional grid.

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Nearest neighbour distributions: New statistical measures for cosmological clustering, 
    [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/staa3604), Volume 500, Issue 4, February 2021, Pages 5479–5499
        
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if QueryPos.shape[1]!=3: 
        raise ValueError('Incorrect spatial dimension for query points: array containing the query point positions must be of shape (n_query, 3), ' \
        'where n_query is the number of query points.')

    if np.any(QueryPos[:, 0] <= 0 or QueryPos[:, 0] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < x < boxsize.')

    if np.any(QueryPos[:, 1] <= 0 or QueryPos[:, 1] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < y < boxsize.')

    if np.any(QueryPos[:, 2] <= 0 or QueryPos[:, 2] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < z < boxsize.')

    if np.any(TracerPos[:, 0] <= 0 or TracerPos[:, 0] >= boxsize):
        raise ValueError('Invalid tracer point position(s): please ensure 0 < x < boxsize.')

    if np.any(TracerPos[:, 1]<= 0 or TracerPos[:, 1]>= boxsize):
        raise ValueError('Invalid tracer point position(s): please ensure 0 < y < boxsize.')

    if np.any(TracerPos[:, 2]<= 0 or TracerPos[:, 2]>= boxsize):
        raise ValueError('Invalid tracer point position(s): please ensure 0 < z < boxsize.')

    if TracerPos.shape[1]!=3: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 3), ' \
        'where n_tracer is the number of tracers.')

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------
        
    #Building the tree
    if Verbose: 
        start_time = time.perf_counter()
        print('\nbuilding the tree ...')
    xtree    = scipy.spatial.cKDTree(TracerPos, boxsize=boxsize)
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the NN distances
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the tracer NN distances ...')
    vol, _ = xtree.query(QueryPos, k=max(kList))[:, np.array(kList)-1]
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the kNN-CDFs
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the tracer auto-CDFs P_{>=k} ...')
    p_gtr_k_list = calc_kNN_CDF(vol, BinsRad)
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------

    #Collecting the results
    if ReturnNNdist:
        kNN_results = (p_gtr_k_list, vol)
    else:
        kNN_results = p_gtr_k_list

    #-----------------------------------------------------------------------------------------------

    if Verbose:
        print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))
    
    return kNN_results

####################################################################################################

def TracerTracerCross3D(boxsize, kA_kB_list, BinsRad, QueryPos, TracerPos_A, TracerPos_B, Verbose=False):
    
    r'''
    Returns the probabilities $P_{\geq k_A}$, $P_{\geq k_B}$ and $P_{\geq k_A, \geq k_B}$ for ($k_A$, $k_B$) in `kA_kB_list` 
    that quantify the extent of the spatial cross-correlation between the given sets of discrete tracers, `TracerPos_A`, `TracerPos_B`.
    	
    1. $P_{\geq k_A}(r)$: 
    	the $k_A$NN-CDF of the first set of discrete tracers, evaluated at radial distance scale $r$
    		
    2. $P_{\geq k_B}(\theta)$: 
    	the $k_B$NN-CDF of the second set of discrete tracers, evaluated at radial distance scale $r$
    		
    3.  $P_{\geq k_A, \geq k_B}(\theta)$:
    	the joint probability of finding at least $k_A$ set A tracers and at least $k_B$ set B tracers within a sphere of radius $r$
    		
    The excess cross-correlation (Banerjee & Abel 2023)[^1] can be computed trivially from the quatities (see the `kNN_ASMR.HelperFunctions.kNN_excess_cross_corr()` method to do this)
    	
    $$\psi_{k_A, k_B} = P_{\geq k_A, \geq k_B}/(P_{\geq k_A} \times P_{\geq k_B})$$
    		
    Parameters
    ----------
    kA_kB_list : list of int tuples
        nearest-neighbour combinations for which the cross-correlations need to be computed (see notes for more details)
    BinsRad : list of numpy float array
        list of radial distance scale arrays (in Mpc/h) for each nearest neighbour combination in `kA_kB_list`. The $i^{th}$ element of the 
        list should contain a numpy array of the desired distances for the $i^{th}$ nearest neighbour combination.
    QueryPos : numpy float array of shape ``(n_query, 3)``
        array of 3D locations for the query points. The 3D locations must be on a grid. The format is (x,y,z) Cartesian coordinates. 
        Please ensure $0<x,y,z<boxsize$.
    TracerPos_A : numpy float array of shape ``(n_tracer, 3)``
        array of 3D locations for the first set of discrete tracers. The 3D locations must be on a grid. The format is (x,y,z) Cartesian coordinates. 
        Please ensure $0<x,y,z<boxsize$.
    TracerPos_B : numpy float array of shape ``(n_tracer, 3)``
        array of 3D locations for the second set of discrete tracers. The 3D locations must be on a grid. The format is (x,y,z) Cartesian coordinates. 
        Please ensure $0<x,y,z<boxsize$.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    p_gtr_kA_list: list of numpy float arrays
        list of auto kNN-CDFs of the first set of discrete tracers evaluated at the desired distance bins. The $i^{th}$ element represents the $k_A^i$NN-CDF, where the $i^{th}$ element of `kA_kB_list` is ($k_A^i$, $k_B^i$).
        
    p_gtr_kB_list: list of numpy float arrays
        list of auto kNN-CDFs of the second set of discrete tracers evaluated at the desired distance bins. The $i^{th}$ element represents the $k_B^i$NN-CDF, where the $i^{th}$ element of `kA_kB_list` is ($k_A^i$, $k_B^i$).
    
    p_gtr_kA_kB_list: list of numpy float arrays
        list of joint tracer-tracer nearest neighbour distributions evaluated at the desired distance bins. The $i^{th}$ element represents the joint {$k_A^i$, $k_B^i$}NN-CDF, where the $i^{th}$ element of `kA_kB_list` is ($k_A^i$, $k_B^i$).
        
    Raises
    ------
    ValueError
        if the lengths of `BinsRad` and `kA_kB_list` do not match.
    ValueError
        if the given query points are not on a three-dimensional grid.
    ValueError
        if x,y, or z coordinates of any of the query points is not in ``(0, boxsize)``.
    ValueError
        if x,y, or z coordinates of any of the tracer points is not in ``(0, boxsize)``.
    ValueError
        if any of the given tracer points are not on a three-dimensional grid.

    Notes
    -----
    The parameter `kA_kB_list` should provide the desired combinations of NN indices for the two tracers sets being cross-correlated. For example, if you wish to compute the joint {1,1}, {1,2} and {2,1}NN-CDFs, then set
            
        kA_kB_list = [(1,1), (1,2), (2,1)]

    Please note that if the number density of one set of tracers is significantly smaller than the other, the joint kNN-CDFs approach the auto kNN-CDFs of the less dense tracer set. In this scenario, it may be better to treat the denser tracer set as a continuous field and use the `TracerFieldCross2DA()` method instead to conduct the cross-correlation analysis  (see Gupta & Banerjee (2024)[^2] for a detailed discussion).

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Cosmological cross-correlations and nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stab961), Volume 504, Issue 2, June 2021, Pages 2911–2923
        
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if len(BinsRad)!=len(kA_kB_list): 
        raise ValueError("length of 'BinsRad' must match length of 'kA_kB_list'.")

    if QueryPos.shape[1]!=3: 
        raise ValueError('Incorrect spatial dimension for query points: array containing the query point positions must be of shape (n_query,3),' \
        ' where n_query is the number of query points.')
    
    if np.any(QueryPos[:, 0] <= 0 or QueryPos[:, 0] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < x < boxsize.')

    if np.any(QueryPos[:, 1] <= 0 or QueryPos[:, 1] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < y < boxsize.')

    if np.any(QueryPos[:, 2] <= 0 or QueryPos[:, 2] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < z < boxsize.')

    if np.any(TracerPos_A[:, 0] <= 0 or TracerPos_A[:, 0] >= boxsize):
        raise ValueError('Invalid tracer point position(s) for the first set: please ensure 0 < x < boxsize.')

    if np.any(TracerPos_A[:, 1]<= 0 or TracerPos_A[:, 1]>= boxsize):
        raise ValueError('Invalid tracer point position(s) for the first set: please ensure 0 < y < boxsize.')

    if np.any(TracerPos_A[:, 2]<= 0 or TracerPos_A[:, 2]>= boxsize):
        raise ValueError('Invalid tracer point position(s) for the first set: please ensure 0 < z < boxsize.')

    if np.any(TracerPos_B[:, 0] <= 0 or TracerPos_B[:, 0] >= boxsize):
        raise ValueError('Invalid tracer point position(s) for the second set: please ensure 0 < x < boxsize.')

    if np.any(TracerPos_A[:, 1]<= 0 or TracerPos_A[:, 1]>= boxsize):
        raise ValueError('Invalid tracer point position(s) for the second set: please ensure 0 < y < boxsize.')

    if np.any(TracerPos_A[:, 2]<= 0 or TracerPos_A[:, 2]>= boxsize):
        raise ValueError('Invalid tracer point position(s) for the second set: please ensure 0 < z < boxsize.')

    
    if TracerPos_A.shape[1]!=3 or TracerPos_B.shape[1]!=3: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 3),' \
        ' where n_tracer is the number of tracers.')

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #Figuring out the NN indices from the kA_kB_list
    kList_A, kList_B = [], []
    for kA, kB in kA_kB_list:
        kList_A.append(kA)
        kList_B.append(kB)
    kMax_A, kMax_B = max(kList_A), max(kList_B)

    #-----------------------------------------------------------------------------------------------
        
    #Building the trees
    if Verbose: 
        start_time = time.perf_counter()
        print('\nbuilding the trees ...')
        start_time_A = time.perf_counter()
    xtree_A = scipy.spatial.cKDTree(TracerPos_A, boxsize=boxsize)
    if Verbose: 
        print('\tfirst set of tracers done; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_A))
        start_time_B = time.perf_counter()
    xtree_B = scipy.spatial.cKDTree(TracerPos_B, boxsize=boxsize)
    if Verbose: 
        print('\tsecond set of tracers done; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_B))
        print('\tcombined time: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the NN distances
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the tracer NN distances ...')
    vol_A, _ = xtree_A.query(QueryPos, k=kMax_A)
    vol_B, _ = xtree_B.query(QueryPos, k=kMax_B)
    req_vol_A, _ = vol_A[:, np.array(kList_A)-1]
    req_vol_B, _ = vol_B[:, np.array(kList_B)-1]
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the auto kNN-CDFs
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the tracer auto-CDFs P_{>=kA}, P_{>=kB} ...')
    p_gtr_kA_list = calc_kNN_CDF(req_vol_A, BinsRad)
    p_gtr_kB_list = calc_kNN_CDF(req_vol_B, BinsRad)
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------

    #Calculating the joint kNN-CDFs
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the joint-CDFs P_{>=kA, >=kB} ...')
    joint_vol = np.zeros((vol_A.shape, len(kA_kB_list)))
    for i, _ in enumerate(kA_kB_list):
        joint_vol[:, i] = np.maximum(req_vol_A[:, i], req_vol_B[:, i])
    p_gtr_kA_kB_list = calc_kNN_CDF(joint_vol, BinsRad)
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------

    if Verbose:
        print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))
    
    return p_gtr_kA_list, p_gtr_kB_list, p_gtr_kA_kB_list

####################################################################################################
