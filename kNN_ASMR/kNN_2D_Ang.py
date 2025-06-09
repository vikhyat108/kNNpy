####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
from sklearn.neighbors import BallTree
import time
import sys
import os

#Necessary for relative imports (see https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im)
module_path = os.path.abspath(os.path.join(''))
'''
@private
'''

if module_path not in sys.path:
    sys.path.append(module_path)

#Importing the required helper function
from kNN_ASMR.HelperFunctions import calc_kNN_CDF

####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

def TracerAuto2DA(kList, BinsRad, MaskedQueryPosRad, MaskedTracerPosRad, ReturnNNdist=False,Verbose=False):
    
    r'''
    Computes the $k$NN-CDFs in 2D angular coordinates (Banerjee & Abel (2021)[^1], Gupta & Banerjee (2024)[^2]) of the provided discrete tracer set (`MaskedTracerPosRad`), evaluated at the provided angular distance scales `BinsRad`, for all $k$ in `kList`. Each $k$NN-CDF measures the probability $P_{\geq k}(\theta)$ of finding at least $k$ tracers in a randomly placed spherical cap of radius $\theta$. The $k$NN-CDFs quantify the spatial clustering of the tracers.
    		
    Parameters
    ----------
    kList : int
        the list of nearest neighbours to calculate the distances to. For example, if ``kList = [1, 2, 4]``, the first, second and fourth-nearest neighbour distributions will be computed.
    BinsRad : list of numpy float array
        list of angular distance arrays (in radians) for each nearest neighbour. The $i^{th}$ element of the list should contain a numpy array of the desired distances for the nearest neighbour specified by the $i^{th}$ element of `kList`.
    MaskedQueryPosRad : numpy float array of shape ``(n_query, 2)``
        array of sky locations for the query points. The sky locations must be on a grid. For each query point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    MaskedTracerPosRad : numpy float array of shape ``(n_tracer, 2)``
        array of sky locations for the discrete tracers. For each data point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    ReturnNNdist : bool, optional
        if set to ``True``, the sorted arrays of NN distances will be returned along with the $k$NN-CDFs, by default ``False``.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    kNN_results: tuple of lists or list of numpy float arrays
        results of the kNN computation. If `ReturnNNdist` is ``True``, returns the tuple ``(p_gtr_k_list, vol)`` where `p_gtr_k_list` is the list of auto kNN-CDFs, and `vol` is the list of NN distances. If `ReturnNNdist` is ``False``, returns `p_gtr_k_list` only
        
    Raises
    ------
    ValueError
        if the given query points are not on a two-dimensional grid.
    ValueError
        if declination of any of the query points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the query points is not in ``[0, 2*np.pi]``.
    ValueError
        if declination of any of the tracer points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the tracer points is not in ``[0, 2*np.pi]``.
    ValueError
        if the given tracer points are not on a two-dimensional grid.

    Notes
    -----
    Data with associated observational footprints are supported, in which case, only tracer positions within the footprint should be provided.Importantly, in this case, query points need to be within the footprint and appropriately padded from the edges of the footprint (see Gupta & Banerjee (2024)[^2] for a detailed discussion). See the `kNN_ASMR.HelperFunctions.create_query_2DA()` method for help with masking and creating the modified query positions.

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Nearest neighbour distributions: New statistical measures for cosmological clustering, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/staa3604), Volume 500, Issue 4, February 2021, Pages 5479–5499
        
    [^2]: Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if MaskedQueryPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for query points: array containing the query point positions must be of shape (n_query, 2), where n_query is the number of query points.')

    if np.any(MaskedQueryPosRad[:, 0]<-np.pi/2 or MaskedQueryPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid query point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedQueryPosRad[:, 1]<0 or MaskedQueryPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid query point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if np.any(MaskedTracerPosRad[:, 0]<-np.pi/2 or MaskedTracerPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid tracer point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedTracerPosRad[:, 1]<0 or MaskedTracerPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid tracer point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if MaskedTracerPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 2), where n_tracer is the number of tracers.')

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------
        
    #Building the tree
    if Verbose: 
        start_time = time.perf_counter()
        print('\nbuilding the tree ...')
    xtree = BallTree(MaskedTracerPosRad, metric='haversine')
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the NN distances
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the tracer NN distances ...')
    vol, _ = xtree.query(MaskedQueryPosRad, k=max(kList))[:, np.array(kList)-1]
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

def TracerTracerCross2DA(kA_kB_list, BinsRad, MaskedQueryPosRad, MaskedTracerPosRad_A, MaskedTracerPosRad_B, Verbose=False):
    
    r'''
    Returns the probabilities $P_{\geq k_A}$, $P_{\geq k_B}$ and $P_{\geq k_A, \geq k_B}$ for ($k_A$, $k_B$) in `kA_kB_list` that quantify the extent of the spatial cross-correlation between the given sets of discrete tracers, `MaskedTracerPosRad_A`, `MaskedTracerPosRad_B`.
    	
    1. $P_{\geq k_A}(\theta)$: 
    	the $k_A$NN-CDF of the first set of discrete tracers, evaluated at angular distance scale $\theta$
    		
    2. $P_{\geq k_B}(\theta)$: 
    	the $k_B$NN-CDF of the second set of discrete tracers, evaluated at angular distance scale $\theta$
    		
    3.  $P_{\geq k_A, \geq k_B}(\theta)$:
    	the joint probability of finding at least $k_A$ set A tracers and at least $k_B$ set B tracers within a spherical cap of radius $\theta$
    		
    The excess cross-correlation (Banerjee & Abel 2023)[^1] can be computed trivially from the quatities (see the `kNN_ASMR.HelperFunctions.kNN_excess_cross_corr()` method to do this)
    	
    $$\psi_{k_A, k_B} = P_{\geq k_A, \geq k_B}/(P_{\geq k_A} \times P_{\geq k_B})$$
    		
    Parameters
    ----------
    kA_kB_list : list of int tuples
        nearest-neighbour combinations for which the cross-correlations need to be computed (see notes for more details)
    BinsRad : list of numpy float array
        list of angular distance scale arrays (in radians) for each nearest neighbour combination in `kA_kB_list`. The $i^{th}$ element of the list should contain a numpy array of the desired distances for the $i^{th}$ nearest neighbour combination.
    MaskedQueryPosRad : numpy float array of shape ``(n_query, 2)``
        array of sky locations for the query points. The sky locations must be on a grid. For each query point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    MaskedTracerPosRad_A : numpy float array of shape ``(n_tracer_A, 2)``
        array of sky locations for the first set of discrete tracers. For each data point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    MaskedTracerPosRad_B : numpy float array of shape ``(n_tracer_B, 2)``
        array of sky locations for the second set of discrete tracers. For each data point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
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
        if the given query points are not on a two-dimensional grid.
    ValueError
        if declination of any of the query points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the query points is not in ``[0, 2*np.pi]``.
    ValueError
        if declination of any of the tracer points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the tracer points is not in ``[0, 2*np.pi]``.
    ValueError
        if any of the given tracer points are not on a two-dimensional grid.

    See Also
    --------
    kNN_ASMR.kNN_2D_Ang.TracerFieldCross2DA : computes tracer-field cross-correlations using the $k$NN formalism.

    Notes
    -----
    Measures the angular cross-correlation between two sets of discrete tracers using the k-nearest neighbour (kNN) formalism as defined in Banerjee & Abel (2021)[^1].

    Data with associated observational footprints are supported, in which case, only tracer positions within the net footprint should be provided (if the two tracer sets have different footprints, a net footprint representing the intersection of the two footprints should be used). Importantly, in this case, query points need to be within the net footprint and appropriately padded from the edges of the footprint (see Gupta & Banerjee (2024)[^2] for a detailed discussion). Please refer to the `kNN_ASMR.HelperFunctions.create_query_2DA()` method for help with masking and creating the modified query positions.

    The parameter `kA_kB_list` should provide the desired combinations of NN indices for the two tracers sets being cross-correlated. For example, if you wish to compute the joint {1,1}, {1,2} and {2,1}NN-CDFs, then set
            
        kA_kB_list = [(1,1), (1,2), (2,1)]

    Please note that if the number density of one set of tracers is significantly smaller than the other, the joint kNN-CDFs approach the auto kNN-CDFs of the less dense tracer set. In this scenario, it may be better to treat the denser tracer set as a continuous field and use the `TracerFieldCross2DA()` method instead to conduct the cross-correlation analysis  (see Gupta & Banerjee (2024)[^2] for a detailed discussion).

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Cosmological cross-correlations and nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stab961), Volume 504, Issue 2, June 2021, Pages 2911–2923
        
    [^2]: Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if len(BinsRad)!=len(kA_kB_list): 
        raise ValueError("length of 'BinsRad' must match length of 'kA_kB_list'.")

    if MaskedQueryPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for query points: array containing the query point positions must be of shape (n_query, 2), where n_query is the number of query points.')

    if np.any(MaskedQueryPosRad[:, 0]<-np.pi/2 or MaskedQueryPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid query point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedQueryPosRad[:, 1]<0 or MaskedQueryPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid query point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if np.any(MaskedTracerPosRad_A[:, 0]<-np.pi/2 or MaskedTracerPosRad_A[:, 0]>np.pi/2 or MaskedTracerPosRad_B[:, 0]<-np.pi/2 or MaskedTracerPosRad_B[:, 0]>np.pi/2):
        raise ValueError('Invalid tracer point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedTracerPosRad_A[:, 1]<0 or MaskedTracerPosRad_A[:, 0]>2*np.pi or MaskedTracerPosRad_B[:, 1]<0 or MaskedTracerPosRad_B[:, 0]>2*np.pi):
        raise ValueError('Invalid tracer point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if MaskedTracerPosRad_A.shape[1]!=2 or MaskedTracerPosRad_B.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 2), where n_tracer is the number of tracers.')

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
    xtree_A = BallTree(MaskedTracerPosRad_A, metric='haversine')
    if Verbose: 
        print('\tfirst set of tracers done; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_A))
        start_time_B = time.perf_counter()
    xtree_B = BallTree(MaskedTracerPosRad_B, metric='haversine')
    if Verbose: 
        print('\tsecond set of tracers done; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_B))
        print('\tcombined time: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the NN distances
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the tracer NN distances ...')
    vol_A, _ = xtree_A.query(MaskedQueryPosRad, k=kMax_A)
    vol_B, _ = xtree_B.query(MaskedQueryPosRad, k=kMax_B)
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

def TracerTracerCross2DA_DataVector(kA_kB_list, BinsRad, MaskedQueryPosRad, MaskedTracerPosVectorRad_A, MaskedTracerPosRad_B, Verbose=False):
    
    r'''
    Returns 'data vectors' of the probabilities $P_{\geq k_A}$, $P_{\geq k_B}$ and $P_{\geq k_A, \geq k_B}$ [refer to kNN_ASMR.kNN_2D_Ang.TracerTracerCross2DA for definitions] for ($k_A$, $k_B$) in `kA_kB_list` for multiple realisations of discrete tracer A [`MaskedTracerPosVectorRad_A`] and a single realisation of the discrete tracer `MaskedTracerPosRad_B`. Please refer to notes to understand why this might be useful.
    	
    Parameters
    ----------
    kA_kB_list : list of int tuples
        nearest-neighbour combinations for which the cross-correlations need to be computed (see notes for more details)
    BinsRad : list of numpy float array
        list of angular distance scale arrays (in radians) for each nearest neighbour combination in `kA_kB_list`. The $i^{th}$ element of the list should contain a numpy array of the desired distances for the $i^{th}$ nearest neighbour combination.
    MaskedQueryPosRad : numpy float array of shape ``(n_query, 2)``
        array of sky locations for the query points. The sky locations must be on a grid. For each query point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    MaskedTracerPosVectorRad_A : numpy float array of shape ``(n_realisations, n_tracer_A, 2)``
        array of sky locations for the first set of discrete tracers. For each data point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    MaskedTracerPosRad_B : numpy float array of shape ``(n_tracer_B, 2)``
        array of sky locations for the second set of discrete tracers. For each data point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    p_gtr_kA_list: list of numpy float arrays
        list of auto kNN-CDFs of the first set of discrete tracers evaluated at the desired distance bins. The $i^{th}$ element is a 2D array of shape ``(n_realisations, n_bins)`` containing the measured $k_A^i$NN-CDFs, where the $i^{th}$ element of `kA_kB_list` is ($k_A^i$, $k_B^i$).
        
    p_gtr_kB_list: list of numpy float arrays
        list of auto kNN-CDFs of the second set of discrete tracers evaluated at the desired distance bins. The $i^{th}$ element represents the $k_B^i$NN-CDF, where the $i^{th}$ element of `kA_kB_list` is ($k_A^i$, $k_B^i$).
    
    p_gtr_kA_kB_list: list of numpy float arrays
        list of joint tracer-tracer nearest neighbour distributions evaluated at the desired distance bins. The $i^{th}$ element is a 2D array of shape ``(n_realisations, n_bins)`` containing the measured joint {$k_A^i$, $k_B^i$}NN-CDF, where the $i^{th}$ element of `kA_kB_list` is ($k_A^i$, $k_B^i$).
        
    Raises
    ------
    ValueError
        if the lengths of `BinsRad` and `kA_kB_list` do not match.
    ValueError
        if the given query points are not on a two-dimensional grid.
    ValueError
        if declination of any of the query points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the query points is not in ``[0, 2*np.pi]``.
    ValueError
        if declination of any of the tracer points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the tracer points is not in ``[0, 2*np.pi]``.
    ValueError
        if any of the given tracer points are not on a two-dimensional grid.

    See Also
    --------
    kNN_ASMR.kNN_2D_Ang.TracerTracerCross2DA: computes tracer-tracer cross-correlation for a single realisation of both tracers using the $k$NN formalism.
    
    kNN_ASMR.kNN_2D_Ang.TracerFieldCross2DA_DataVector : computes tracer-field cross-correlation data vectors for multiple realisations of the tracer using the $k$NN formalism.

    Notes
    -----
    Please refer to the documentation of kNN_ASMR.kNN_2D_Ang.TracerTracerCross2DA for important usage notes that also apply to this function. <Explain why cross-correlating multiple realisations of tracer A with single realisation of tracer B might be useful>
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if len(BinsRad)!=len(kA_kB_list): 
        raise ValueError("length of 'BinsRad' must match length of 'kA_kB_list'.")

    if MaskedQueryPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for query points: array containing the query point positions must be of shape (n_query, 2), where n_query is the number of query points.')

    if np.any(MaskedQueryPosRad[:, 0]<-np.pi/2 or MaskedQueryPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid query point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedQueryPosRad[:, 1]<0 or MaskedQueryPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid query point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if np.any(MaskedTracerPosRad_A[:, :, 0]<-np.pi/2 or MaskedTracerPosRad_A[:, :, 0]>np.pi/2 or MaskedTracerPosRad_B[:, 0]<-np.pi/2 or MaskedTracerPosRad_B[:, 0]>np.pi/2):
        raise ValueError('Invalid tracer point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedTracerPosRad_A[:, :, 1]<0 or MaskedTracerPosRad_A[:, :, 0]>2*np.pi or MaskedTracerPosRad_B[:, 1]<0 or MaskedTracerPosRad_B[:, 0]>2*np.pi):
        raise ValueError('Invalid tracer point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if MaskedTracerPosRad_A.shape[2]!=2 or MaskedTracerPosRad_B.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 2), where n_tracer is the number of tracers.')

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #Figuring out the NN indices from the kA_kB_list
    kList_A, kList_B = [], []
    for kA, kB in kA_kB_list:
        kList_A.append(kA)
        kList_B.append(kB)
    kMax_A, kMax_B = max(kList_A), max(kList_B)

    #-----------------------------------------------------------------------------------------------
        
    #Nearest-neighbour computations for tracer B. This is done outside the loop over realisations of tracer A to avoid unnecessary repetition

    if Verbose: print('\n\nFirst carrying out nearest-neighbour computations for tracer B\n')

    #Building the tree
    if Verbose: 
        start_time = time.perf_counter()
        print('\nbuilding the tree...')
    xtree_B = BallTree(MaskedTracerPosRad_B, metric='haversine')
    if Verbose: 
        print('\ttime taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #Calculating the NN distances
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the NN distances...')
    vol_B, _ = xtree_B.query(MaskedQueryPosRad, k=kMax_B)
    req_vol_B, _ = vol_B[:, np.array(kList_B)-1]
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #Calculating the auto kNN-CDFs
    if Verbose: 
        start_time = time.perf_counter()
        print('\ncomputing the auto-CDFs P_{>=kB} ...')
    p_gtr_kB_list = calc_kNN_CDF(req_vol_B, BinsRad)
    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------

    #Now looping over the multiple realisations of tracer A

    if Verbose: print('\n\nNow looping over the realisations of tracer A\n')

    n_reals = MaskedTracerPosVectorRad_A.shape[0]
    p_gtr_kA_veclist, p_gtr_kA_kB_veclist = [], [], []
    for k_ind in range(len(kA_kB_list)):
        p_gtr_kA_veclist.append(np.zeros((n_reals, len(BinsRad[k_ind]))))
        p_gtr_kA_kB_veclist.append(np.zeros((n_reals, len(BinsRad[k_ind]))))

    for realisation, MaskedTracerPosRad_A in enumerate(MaskedTracerPosVectorRad_A):

        if Verbose: 
            start_time = time.perf_counter()
            print(f'\n\n--------------  Realisation {realisation+1}/{n_reals}  --------------\n')

        #Building the tree
        if Verbose: 
            start_time_tree = time.perf_counter()
            print('\nbuilding the trees ...')
        xtree_A = BallTree(MaskedTracerPosRad_A, metric='haversine')
        if Verbose: 
            print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_tree))

        #Calculating the NN distances
        if Verbose: 
            start_time_NN = time.perf_counter()
            print('\ncomputing the NN distances ...')
        vol_A, _ = xtree_A.query(MaskedQueryPosRad, k=kMax_A)
        req_vol_A, _ = vol_A[:, np.array(kList_A)-1]
        if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_NN))
    
        #Calculating the auto kNN-CDFs
        if Verbose: 
            start_time_CDF = time.perf_counter()
            print('\ncomputing the auto-CDFs P_{>=kA} ...')
        p_gtr_kA_list = calc_kNN_CDF(req_vol_A, BinsRad)
        for k_ind in range(len(kA_kB_list)):
            p_gtr_kA_veclist[k_ind][realisation] = p_gtr_kA_list[k_ind]
        if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_CDF))

        #Calculating the joint kNN-CDFs
        if Verbose: 
            start_time_joint = time.perf_counter()
            print('\ncomputing the joint-CDFs P_{>=kA, >=kB} ...')
        joint_vol = np.zeros((vol_A.shape, len(kA_kB_list)))
        for i, _ in enumerate(kA_kB_list):
            joint_vol[:, i] = np.maximum(req_vol_A[:, i], req_vol_B[:, i])
        p_gtr_kA_kB_list = calc_kNN_CDF(joint_vol, BinsRad)
        for k_ind in range(len(kA_kB_list)):
            p_gtr_kA_kB_veclist[k_ind][realisation] = p_gtr_kA_kB_list[k_ind]
        if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_joint))

        if Verbose: 
            print('\ntime taken for realisation {}: {:.2e} s.'.format(realisation+1, time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------

    if Verbose:
        print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))
    
    return p_gtr_kA_veclist, p_gtr_kB_list, p_gtr_kA_kB_veclist

####################################################################################################

def TracerFieldCross2DA(kList, BinsRad, MaskedQueryPosRad, MaskedTracerPosRad, SmoothedFieldDict,  FieldConstPercThreshold, Verbose=False):
    
    r'''
    Returns the probabilities $P_{\geq k}$, $P_{>{\rm dt}}$ and $P_{\geq k,>{\rm dt}}$ for $k$ in `kList`, that quantify the extent of the spatial cross-correlation between the given discrete tracer positions (`MaskedTracerPosRad`) and the given continuous overdensity field (`SmoothedFieldDict`).
    	
    1. $P_{\geq k}(\theta)$: 
    	the kNN-CDF of the discrete tracers, evaluated at angular distance scale $\theta$
    		
    2. $P_{>{\rm dt}}(\theta)$: 
    	the probability of the overdensity field smoothed with a top-hat filter of angular size $\theta$ exceeding the given constant percentile density threshold
    		
    3. $P_{\geq k, >{\rm dt}}(\theta)$:
    	the joint probability of finding at least 'k' tracers within a spherical cap of radius $\theta$ AND the overdensity field smoothed at angular scale $\theta$ exceeding the given density threshold (as specified by the parameter `FieldConstPercThreshold`)
    		
    The excess cross-correlation (Banerjee & Abel 2023)[^1] can be computed trivially from the quatities (see the `kNN_ASMR.HelperFunctions.kNN_excess_cross_corr()` method to do this)
    	
    $$\psi_{k, {\rm dt}} = P_{\geq k, >{\rm dt}}/(P_{\geq k} \times P_{>{\rm dt}})$$

    Parameters
    ----------
    kList : int
        the list of nearest neighbours to calculate the distances to. For example, if ``kList = [1, 2, 4]``, the first, second and fourth-nearest neighbour distributions will be computed.
    BinsRad : list of numpy float array
        list of angular distance arrays (in radians) for each nearest neighbour. The $i^{th}$ element of the list should contain a numpy array of the desired distances for the nearest neighbour specified by the $i^{th}$ element of `kList`.
    MaskedQueryPosRad : numpy float array of shape ``(n_query, 2)``
        array of sky locations for the query points. The sky locations must be on a grid. For each query point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    MaskedTracerPosRad : numpy float array of shape ``(n_tracer, 2)``
        array of sky locations for the discrete tracers. For each data point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    SmoothedFieldDict : dict
        dictionary containing the continuous field masked within the observational footprint and smoothed at various angular distance scales. For example, ``SmoothedFieldDict['0.215']`` represents the continuous map smoothed at a scale of 0.215 radians.
    FieldConstPercThreshold : float
        the percentile value for the constant percentile threshold to be used for the continuous field. For example, ``FieldConstPercThreshold = 75.0`` represents a 75th percentile threshold.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    p_gtr_k_list: list of numpy float arrays
        auto kNN-CDFs of the discrete tracers evaluated at the desired distance bins.
        
    p_gtr_dt_list: list of numpy float arrays
        continuum version of auto kNN-CDFs for the continuous field evaluated at the desired distance bins.
    
    p_gtr_k_dt_list: list of numpy float arrays
        joint tracer-field nearest neighbour distributions evaluated at the desired distance bins.

    Raises
    ------
    ValueError
        if the given query points are not on a two-dimensional grid.
    ValueError
        if declination of any of the query points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the query points is not in ``[0, 2*np.pi]``.
    ValueError
        if declination of any of the tracer points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the tracer points is not in ``[0, 2*np.pi]``.
    ValueError
        if the given tracer points are not on a two-dimensional grid.
    ValueError
        if the shape of field smoothed at a particular scale does not match the shape of the query point array.

    See Also
    --------
    kNN_ASMR.kNN_2D_Ang.TracerTracerCross2DA : computes tracer-tracer cross-correlations using the $k$NN formalism.

    Notes
    -----
    Measures the angular cross-correlation between a set of discrete tracers and a continuous overdensity field using the k-nearest neighbour (kNN) formalism as defined in Banerjee & Abel (2023)[^1] and Gupta & Banerjee (2024)[^2].
    
    The field must already be smoothed at the desired angular distance scales using a top-hat filter (see the `kNN_ASMR.HelperFunctions` module for help with smoothing). The smoothed fields need to be provided as a dictionary (`SmoothedFieldDict`), see below for further details.
    
    Currently, the algorithm requires a constant percentile overdensity threshold for the continuous field and query points to be defined on a HEALPix grid. Extentions to a constant mass threshold and poisson-sampled query points on the sky may be added in the future. 
    
    Data with associated observational footprints are supported, in which case, only tracer positions within the footprint should be provided and the field should be masked appropriately.Importantly, in this case, query points need to be within the footprint and appropriately padded from the edges of the footprint (see Gupta & Banerjee (2024)[^2] for a detailed discussion). If the footprints of the tracer set and the field are different, a combined mask representing the intersection of the two footprints should be used (see the `kNN_ASMR.HelperFunctions.create_query_2DA()` method for help with masking and creating the modified query positions).

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour   distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stac3813), Volume 519, Issue 4, March 2023, Pages 4856–4868
        
    [^2]: Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if MaskedQueryPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for query points: array containing the query point positions must be of shape (n_query, 2), where n_query is the number of query points.')

    if np.any(MaskedQueryPosRad[:, 0]<-np.pi/2 or MaskedQueryPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid query point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedQueryPosRad[:, 1]<0 or MaskedQueryPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid query point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if np.any(MaskedTracerPosRad[:, 0]<-np.pi/2 or MaskedTracerPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid tracer point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedTracerPosRad[:, 1]<0 or MaskedTracerPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid tracer point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if MaskedTracerPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 2), where n_tracer is the number of tracers.')

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------
        
    #Step 1: Calculate nearest neighbour distances of query points, and the kNN-CDFs for the discrete tracers

    if Verbose: 
        step_1_start_time = time.perf_counter()
        print('\ninitiating step 1 ...')

    #-----------------------------------------------------------------------------------------------
    
    #Building the tree
    if Verbose: 
        start_time = time.perf_counter()
        print('\n\tbuilding the tree ...')
    xtree = BallTree(MaskedTracerPosRad, metric='haversine')
    if Verbose: print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the NN distances
    if Verbose: 
        start_time = time.perf_counter()
        print('\n\tcomputing the tracer NN distances ...')
    vol, _ = xtree.query(MaskedQueryPosRad, k=max(kList))[:, np.array(kList)-1]
    if Verbose: print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the kNN-CDFs
    if Verbose: 
        start_time = time.perf_counter()
        print('\n\tcomputing the tracer auto-CDFs P_{>=k} ...')
    p_gtr_k_list = calc_kNN_CDF(vol, BinsRad)

    #-----------------------------------------------------------------------------------------------
    
    if Verbose: 
        print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))
        print('\ntime taken for step 1: {:.2e} s.'.format(time.perf_counter()-step_1_start_time))
    
    #-----------------------------------------------------------------------------------------------
        
    #Steps 2: calculate the fraction of query points with nearest neighbour distance less than the angular distance and smoothened field greater than the overdensity threshold

    if Verbose: 
        step_2_start_time = time.perf_counter()
        print('\ninitiating step 2 ...')

    #-----------------------------------------------------------------------------------------------
        
    p_gtr_k_dt_list = []
    p_gtr_dt_list = []
    
    for k_ind, k in enumerate(kList):

        if Verbose: 
            start_time = time.perf_counter()
            print('\n\tComputing P_{>dt} and P_{>=k, >dt} for k = {} ...'.format(k))

        p_gtr_k_dt = np.zeros(len(BinsRad[k]))
        p_gtr_dt = np.zeros(len(BinsRad[k]))

        for i, ss in enumerate(BinsRad[k]):

            #---------------------------------------------------------------------------------------

            #Load the smoothed field
            SmoothedField = SmoothedFieldDict[str(ss)]

            #Check if the masked smoothed field is consistent with the number of query points
            if SmoothedField.shape[0]!=MaskedQueryPosRad.shape[0]:
                raise ValueError('Shape of field smoothed at scale {:.3e} rad does not match shape of query point array.'.format(ss))

            #---------------------------------------------------------------------------------------
            
            #Compute the overdensity threshold            
            delta_star_ss = np.percentile(SmoothedField, FieldConstPercThreshold)    

            #---------------------------------------------------------------------------------------

            #Compute the fraction of query points satisfying the joint condition
            ind_gtr_k_dt = np.where((vol[:, k_ind]<ss)&(SmoothedField>delta_star_ss))
            p_gtr_k_dt[i] = len(ind_gtr_k_dt[0])/MaskedQueryPosRad.shape[0]

            #---------------------------------------------------------------------------------------

            #Compute the fraction of query points with smoothed field exceeding the overdensity threshold
            ind_gtr_dt = np.where(SmoothedField>delta_star_ss)
            p_gtr_dt[i] = len(ind_gtr_dt[0])/MaskedQueryPosRad.shape[0]

        #-------------------------------------------------------------------------------------------

        p_gtr_k_dt_list.append(p_gtr_k_dt)
        p_gtr_dt_list.append(p_gtr_dt)

        #-------------------------------------------------------------------------------------------

        if Verbose: print('\t\tdone; time taken for step 2: {:.2e} s.'.format(time.perf_counter()-step_2_start_time))

    #-----------------------------------------------------------------------------------------------

    if Verbose: print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))
    
    return p_gtr_k_list, p_gtr_dt_list, p_gtr_k_dt_list

####################################################################################################

#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################
