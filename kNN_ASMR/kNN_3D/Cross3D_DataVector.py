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

def TracerTracerCross3D_DataVector(boxsize, kA_kB_list, BinsRad, QueryPos, TracerPos_A_dict, TracerPos_B, Verbose=False ):
    r'''
    Returns the probabilities $P_{\geq k_{A_i}}$, $P_{\geq k_B}$ and $P_{\geq k_{A_i}, \geq k_B}$ for ($k_{A_i}$, $k_B$) in `kA_kB_list` for various 
    realizations of Tracer A, while keeping the set Tracer B constant. Refer to Notes to understand why this might be useful. These quantify
    the extent of the spatial cross-correlation between the given sets of discrete tracers, the $i^{\text{th}}$ realization of `TracerPos_A`, `TracerPos_B`.
    We do not vary the 'kA_kB_list' as a function of the realizations of Tracer A.
    	
    1. $P_{\geq k_{A_i}}(r)$: 
    	the $k_A$NN-CDF of the $i^{\text{th}}$ realization of the first set of discrete tracers, evaluated at radial distance scale $r$
    		
    2. $P_{\geq k_B}(\theta)$: 
    	the $k_B$NN-CDF of the second set of discrete tracers, evaluated at radial distance scale $r$
    		
    3.  $P_{\geq k_{A_i}, \geq k_B}(\theta)$:
    	the joint probability of finding at least $k_A$ set A tracers and at least $k_B$ set B tracers within a sphere of radius $r$, for the
        $i^{\text{th}}$ realization of Tracer A
    		
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
    TracerPos_A_dict : dictionary, where each key corresponds to the realization, and stores the corresponding numpy array of size ``(n_tracer,3)``, that 
        is the tracer positions for the $i^{\text{th}}$ realization 
        The 3D locations must be on a grid. The format is (x,y,z) Cartesian coordinates. 
        Please ensure $0<x,y,z<boxsize$.
    TracerPos_B : numpy float array of shape ``(n_tracer, 3)``
        array of 3D locations for the second set of discrete tracers. The 3D locations must be on a grid. The format is (x,y,z) Cartesian coordinates. 
        Please ensure $0<x,y,z<boxsize$.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    Realizations: a numpy array of arrays where the $i^{\text{th}}$ element corresponds to the 3D cross-correlation calculated between the $i^{\text{th}} 
    realization of Tracer A and Tracer B. The values correspond to an numpy array: [p_gtr_kA_list, p_gtr_kB_list, p_gtr_kA_kB_list]
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
    #Write why this module might be useful

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Cosmological cross-correlations and nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stab961), Volume 504, Issue 2, June 2021, Pages 2911–2923
        
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()
    keys=TracerPos_A_dict.keys()

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
    for i in range(len(keys)):
        if np.any(TracerPos_A_dict[keys[i]][:, 0] <= 0 or TracerPos_A_dict[i][:, 0] >= boxsize):
            raise ValueError('Invalid tracer point position(s) for the first set: please ensure 0 < x < boxsize.')

    for i in range(len(keys)):
        if np.any(TracerPos_A_dict[keys[i]][:, 1]<= 0 or TracerPos_A_dict[keys[i]][:, 1]>= boxsize):
            raise ValueError('Invalid tracer point position(s) for the first set: please ensure 0 < y < boxsize.')

    for i in range(len(keys)):
        if np.any(TracerPos_A_dict[keys[i]][:, 2]<= 0 or TracerPos_A_dict[keys[i]][:, 2]>= boxsize):
            raise ValueError('Invalid tracer point position(s) for the first set: please ensure 0 < z < boxsize.')

    if np.any(TracerPos_B[:, 0] <= 0 or TracerPos_B[:, 0] >= boxsize):
        raise ValueError('Invalid tracer point position(s) for the second set: please ensure 0 < x < boxsize.')

    if np.any(TracerPos_B[:, 1]<= 0 or TracerPos_B[:, 1]>= boxsize):
        raise ValueError('Invalid tracer point position(s) for the second set: please ensure 0 < y < boxsize.')

    if np.any(TracerPos_B[:, 2]<= 0 or TracerPos_B[:, 2]>= boxsize):
        raise ValueError('Invalid tracer point position(s) for the second set: please ensure 0 < z < boxsize.')

    for i in range(len(keys)):
        if TracerPos_A_dict[keys[i]].shape[1]!=3 or TracerPos_B.shape[1]!=3: 
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
        start_time_B = time.perf_counter()
    xtree_B = scipy.spatial.cKDTree(TracerPos_B, boxsize=boxsize)  
    if Verbose: 
        print('\tsecond set of tracers done; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_B))

    #Initializing the containinment array
    #Realizations=np.zeros((len(TracerPos_A_dict.values()),3,len(kA_kB_list)))
    Realizations=[]
    
    for i, values in enumerate(TracerPos_A_dict.values()):
        if Verbose:
            print(f'\n Building the tree for the {i}th relaization of Tracer A')
            start_time_A=time.perf_counter()
        xtree_A=scipy.spatial.cKDTree(values, boxsize=boxsize)
        if Verbose:
            print('\tset of tracers being iterated over done; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_A))
            print('\tcombined time: {:.2e} s.'.format(time.perf_counter()-start_time))

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
        
        Realizations.append([p_gtr_kA_list, p_gtr_kB_list, p_gtr_kA_kB_list])
    Realizations=np.array(Realizations) 
    return Realizations

####################################################################################################
