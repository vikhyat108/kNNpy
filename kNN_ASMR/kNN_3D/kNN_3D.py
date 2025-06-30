import numpy as np
import time
import sys
import scipy.spatial
import os
import gc

# Ensure module path is correctly added for relative imports
module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import necessary helper functions
from kNN_ASMR.HelperFunctions import calc_kNN_CDF
from kNN_ASMR.HelperFunctions import CIC_3D_Interp
from kNN_ASMR.HelperFunctions import smoothing_3D
from kNN_ASMR.HelperFunctions import create_smoothed_field_dict_3D

#################################################################################################################

#----------------------------------------  Function Definitions  ----------------------------------------

def TracerFieldCross3D(kList, RBins, BoxSize, QueryPos, TracerPos, Field3D, FieldConstPercThreshold, ReturnSmoothedFieldDict=False, Verbose=False):
    r'''
    Returns the probabilities $P_{\geq k}$, $P_{>{\rm dt}}$ and $P_{\geq k,>{\rm dt}}$ for $k$ in `kList`, that quantify the extent of the spatial cross-correlation between the given discrete tracer positions (`TracerPos`) and the given continuous overdensity field (`SmoothedFieldDict`) in three-dimensional space.
    
    1. $P_{\geq k}(r)$: 
        the kNN-CDF of the discrete tracers, evaluated at separation $r$
    
    2. $P_{>{\rm dt}}(r)$: 
        the probability of the overdensity field smoothed with a top-hat filter of radius $r$ exceeding the given constant percentile density threshold
    
    3. $P_{\geq k, >{\rm dt}}(r)$:
        the joint probability of finding at least 'k' tracers within a sphere of radius $r$ AND the overdensity field smoothed at scale $r$ exceeding the given density threshold (as specified by the parameter `FieldConstPercThreshold`)
    
    The excess cross-correlation (Banerjee & Abel 2023)[^1] can be computed trivially from these quantities:
    
    $$\psi_{k, {\rm dt}} = \frac{P_{\geq k, >{\rm dt}}}{P_{\geq k} \times P_{>{\rm dt}}}$$

    Parameters
    ----------
    kList : list of int
        List of nearest neighbours to compute. For example, if ``kList = [1, 2, 4]``, the first, second and fourth-nearest neighbour distributions will be computed.

    RBins : list of numpy float arrays
        List of radial distance arrays (in comoving Mpc/$h$), one for each value in `kList`. The i-th element of the list should be a numpy array specifying the distances to be used for the nearest neighbour calculation corresponding to `kList[i]`.

    BoxSize : float
        The size of the cubic box in which the tracers and the continuous field are defined.

    QueryPos : numpy float array of shape ``(n_query, 3)``
        Array of 3D positions (e.g., in Cartesian coordinates) used to query the nearest-neighbour distances, and also compute field's CDF.

    TracerPos : numpy float array of shape ``(n_tracer, 3)``
        Array of 3D positions of discrete tracers, with columns representing the x, y, and z coordinates, respectively.
    
    Field3D : numpy float array of shape ``(n_grid, n_grid, n_grid)``
        A 3D numpy array representing the continuous field (for e.g., the matter overdensity field). The shape of the array should match the grid size used for smoothing.

    FieldConstPercThreshold : float
        The percentile threshold for identifying overdense regions in the continuous field. For example, ``75.0`` indicates the 75th percentile.

    ReturnSmoothedFieldDict : bool, optional
        if set to ``True``, the dictionary containing the continuous field smoothed at the provided radial bins, will be returned along with the nearest-neighbour measurements, by default ``False``.
    
    Verbose : bool, optional
        If True, prints timing information for each step. Default is False.

    Returns
    -------
    p_gtr_k_list : list of numpy float arrays
        Auto kNN-CDFs of the discrete tracers evaluated at the desired distance bins.

    p_gtr_dt_list : list of numpy float arrays
        Overdensity-field auto kNN-CDFs evaluated at the same scales.

    p_gtr_k_dt_list : list of numpy float arrays
        Joint CDFs of finding $\geq k$ tracers AND field value exceeding the threshold at a given scale.

    SmoothedFieldDict : dict
        dictionary containing the continuous field smoothed at the provided radial bins, returned only if `ReturnSmoothedDict` is ``True``. For example, ``SmoothedFieldDict['5']`` represents the continuous map smoothed at a scale of 5 Mpc/h.

    Raises
    ------
    ValueError
        If TracerPos are not 3D.
    ValueError
        If QueryPos are not 3D.
    ValueError
        If tracer positions are outside the specified box size.
    ValueError
        If QueryPos are outside the specified box size.

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour distributions, [MNRAS](https://doi.org/10.1093/mnras/stac3813), Volume 519, Issue 4, March 2023, Pages 4856-4868

    [^2]: Eishica Chand, Arka Banerjee, Simon Foreman, Francisco Villaescusa-Navarro, [MNRAS](https://doi.org/10.1093/mnras/staf433), Volume 538, Issue 3, April 2025, Pages 2204-221 
    '''

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------

    # Step 0: Input validation
    if Verbose: print('Checking inputs ...')

    if QueryPos.shape[1] != 3:
        raise ValueError("Query positions must be 3D (shape: n_query x 3).")
    if TracerPos.shape[1] != 3:
        raise ValueError("Tracer positions must be 3D (shape: n_tracer x 3).")
    if np.any((TracerPos <= 0) | (TracerPos > BoxSize)):
        raise ValueError("Tracer positions must be within the box [0, BoxSize).")

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------
    # Step 1: Compute kNN-CDFs for tracer positions
    if Verbose:
        step_1_start_time = time.perf_counter()
        print('\ninitiating step 1 ...')

    #-----------------------------------------------------------------------------------------------

    # Building the kdTree
    if Verbose:
        print('\n\tbuilding the kdTree ...')
        t_start = time.perf_counter()

    xtree = scipy.spatial.cKDTree(TracerPos, boxsize=BoxSize)

    if Verbose:
        print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter() - t_start))

    #------------------------------------------------------------------------------------------------

    # To store the CDFs for each k  
    if Verbose:
        print('\n\tcomputing the tracer NN distances ...')
        t_start = time.perf_counter()
    

    #-------------------------------------------------------------------------------------------------

    Nquery = QueryPos.shape[0]
    dists, _ = xtree.query(QueryPos, k=max(kList), workers=-1)
    vol = dists[:, np.array(kList)-1]
    
    #------------------------------------------------------------------------------------------------

    # Compute the kNN-CDFs for the tracers
    if Verbose:
        print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter() - t_start))
   
    if Verbose:
        print('\n\tcomputing P_{>=k} ...')
        t_start = time.perf_counter()

    p_gtr_k_list = calc_kNN_CDF(vol, RBins)

    if Verbose:
        print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter() - t_start))
        print('time taken for step 1: {:.2e} s.'.format(time.perf_counter() - step_1_start_time))

    #------------------------------------------------------------------------------------------------

    # Step 2: Compute kNN-CDFs for the overdensity field, and the joint CDFs with tracers 

    if Verbose:
        step_2_start_time = time.perf_counter()
        print('\ninitiating step 2 ...')

    # Store computed smoothed fields, interpolated values, and percentile thresholds
    SmoothedFieldDictOut = {}
    Interpolated_Smoothed_Field = {}
    Delta_Threshold = {}

    # To store the CDFs for each k
    p_gtr_k_dt_list = []
    p_gtr_dt_list = []

    #------------------------------------------------------------------------------------------------

    # Compute the CDFs
    for k_ind, k in enumerate(kList):

        if Verbose:
            print(f"\nComputing P_{{>=k, >dt}} and P_{{>dt}} for k = {k} ...")
            k_start_time = time.perf_counter()

        p_gtr_k_dt = np.zeros(len(RBins[k_ind]))
        p_gtr_dt   = np.zeros(len(RBins[k_ind]))

        for j, ss in enumerate(RBins[k_ind]):

            #------------------------------------------------------------------------------------------------
            ss_str = str(ss)

            if ss_str not in SmoothedFieldDictOut:
                SmoothedFieldDictOut[ss_str] = smoothing_3D(Field3D, Filter='Top-Hat', grid=Field3D.shape[0], BoxSize=BoxSize, R=ss, Verbose=False)

            #-------------------------------------------------------------------------------------------------

            if ss_str not in Interpolated_Smoothed_Field:
                Interpolated_Smoothed_Field[ss_str] = CIC_3D_Interp(QueryPos, SmoothedFieldDictOut[ss_str], BoxSize)

            interp_field = Interpolated_Smoothed_Field[ss_str]

            
            #-------------------------------------------------------------------------------------------------

            if ss_str not in Delta_Threshold:
                Delta_Threshold[ss_str] = np.percentile(Interpolated_Smoothed_Field[ss_str], FieldConstPercThreshold)

            delta_star_ss = Delta_Threshold[ss_str]

            #-------------------------------------------------------------------------------------------------

            # Compute fractions
            vol_mask      = vol[:, k_ind] < ss
            field_mask    = interp_field > delta_star_ss

            p_gtr_dt[j]   = np.count_nonzero(field_mask) / Nquery
            p_gtr_k_dt[j] = np.count_nonzero(vol_mask & field_mask) / Nquery

        #-------------------------------------------------------------------------------------------------

        p_gtr_k_dt_list.append(p_gtr_k_dt)
        p_gtr_dt_list.append(p_gtr_dt)

        if Verbose:
            print(f"\tdone for k = {k}; time taken: {time.perf_counter() - k_start_time:.2e} s")

    #------------------------------------------------------------------------------------------------

    if Verbose:
        print(f"\nTotal time taken: {time.perf_counter() - step_2_start_time:.2e} s")
    
    #-----------------------------------------------------------------------------------------------

    if Verbose:
        print(f"\nTotal time taken for all steps: {time.perf_counter() - total_start_time:.2e} s")

    if ReturnSmoothedFieldDict:
        return p_gtr_k_list, p_gtr_dt_list, p_gtr_k_dt_list, SmoothedFieldDictOut
    else:
        return p_gtr_k_list, p_gtr_dt_list, p_gtr_k_dt_list


#################################################################################################################


def TracerFieldCross3D_DataVector(kList, RBins, BoxSize, QueryPos, TracerPosVector, Field, FieldConstPercThreshold, ReturnSmoothedDict=False, Verbose=False):
    
    r'''
    Returns 'data vectors' of the  the probabilities $P_{\geq k}$, $P_{>{\rm dt}}$ and $P_{\geq k,>{\rm dt}}$ [refer to kNN_ASMR.kNN_3D.TracerFieldCross for definitions] for $k$ in `kList` for multiple realisations of the given discrete tracer set [`TracerPosVector`] and a single realisation of the given continuous overdensity field (`Field`). Please refer to notes to understand why this might be useful.
    	
    Parameters
    ----------
    kList : list of int
        List of nearest neighbours to compute. For example, if ``kList = [1, 2, 4]``, the first, second and fourth-nearest neighbour distributions will be computed.

    RBins : list of numpy float arrays
        List of radial distance arrays (in comoving Mpc/$h$), one for each value in `kList`. The i-th element of the list should be a numpy array specifying the distances to be used for the nearest neighbour calculation corresponding to `kList[i]`.

    BoxSize : float
        The size of the cubic box in which the tracers and the continuous field are defined.

    QueryPos : numpy float array of shape ``(n_query, 3)``
        Array of 3D positions (e.g., in Cartesian coordinates) used to query the nearest-neighbour distances, and also compute field's CDF.
    
    TracerPosVector : numpy float array of shape ``(n_realisations, n_tracer, 3)``
        Array of 3D positions of n_realisations of discrete tracers, with columns representing the x, y, and z coordinates, respectively.
    
    Field : numpy float array of shape ``(n_grid, n_grid, n_grid)``
        A 3D numpy array representing the continuous field (for e.g., the matter overdensity field). The shape of the array should match the grid size used for smoothing.

    FieldConstPercThreshold : float
        The percentile threshold for identifying overdense regions in the continuous field. For example, ``75.0`` indicates the 75th percentile.

    ReturnSmoothedFieldDict : bool, optional
        if set to ``True``, the dictionary containing the continuous field smoothed at the provided radial bins, will be returned along with the nearest-neighbour measurements, by default ``False``.
    
    Verbose : bool, optional
        If True, prints timing information for each step. Default is False.

    Returns
    -------
    p_gtr_k_veclist: list of numpy float arrays
        list of auto kNN-CDFs of the discrete tracers evaluated at the desired distance bins. Each list member is a 2D array of shape ``(n_realisations, n_bins)``.

    p_gtr_dt_list: list of numpy float arrays
        continuum version of auto kNN-CDFs for the continuous field evaluated at the desired distance bins.

    p_gtr_k_dt_veclist: list of numpy float arrays
        list of joint tracer-field nearest neighbour distributions evaluated at the desired distance bins. Each list member is a 2D array of shape ``(n_realisations, n_bins)``.

    SmoothedFieldDict : dict
        dictionary containing the continuous field smoothed at the provided radial bins, returned only if `ReturnSmoothedDict` is ``True``. For example, ``SmoothedFieldDict['5']`` represents the continuous map smoothed at a scale of 5 Mpc/h.

    Raises
    ------
    ValueError
        If TracerPos are not on a 3dimensional grid.
    ValueError
        If QueryPos are not on a 3dimensional grid.
    ValueError
        If tracer positions are outside the specified box size.
    ValueError
        If QueryPos are outside the specified box size.

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour distributions, [MNRAS](https://doi.org/10.1093/mnras/stac3813), Volume 519, Issue 4, March 2023, Pages 4856-4868

    [^2]: Eishica Chand, Arka Banerjee, Simon Foreman, Francisco Villaescusa-Navarro, [MNRAS](https://doi.org/10.1093/mnras/staf433), Volume 538, Issue 3, April 2025, Pages 2204-221 
    '''

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose:
        print('Checking inputs ...')

    # Query positions must be (n_query, 3)
    if QueryPos.ndim != 2 or QueryPos.shape[1] != 3:
        raise ValueError("Query positions must be of shape: (n_query, 3), where n_query is the number of query points.")

    # Tracer positions must be (n_realizations, n_tracer, 3)
    if TracerPosVector.ndim != 3 or TracerPosVector.shape[2] != 3:
        raise ValueError("Tracer positions must be of shape: (n_realizations, n_tracer, 3), where n_realizations is the number of tracer realizations and n_tracer is the number of tracers per realization.")

    # Tracer positions must lie within [0, BoxSize)
    if np.any((TracerPosVector <= 0) | (TracerPosVector > BoxSize)):
        raise ValueError("Tracer positions must be within the box [0, BoxSize).")

    if Verbose:
        print('\tdone.')

    #-----------------------------------------------------------------------------------------------
        
    #Step 1: smooth the continuous field and store in dictionary
    if Verbose:
        step_1_start_time = time.perf_counter()
        print('\ninitiating step 1 (smoothing the continuous field at the given radial scales)...')

    #-----------------------------------------------------------------------------------------------

    grid = Field.shape[0]  
    Filter = 'Top-Hat'

    SmoothedFieldDict = create_smoothed_field_dict_3D(Field, Filter, grid, BoxSize, RBins, thickness=None, Verbose=False)

    if Verbose: print('\tdone; time taken for step 1: {:.2e} s.'.format(time.perf_counter()-step_1_start_time))

    #-----------------------------------------------------------------------------------------------
        
    #Step 2: 

    # A. Compute the fraction of query points at which the smoothed fields at the different radial
    #    scales are greater than the overdensity threshold.

    # B. For each realization of the discrete tracer set, calculate 
    #   (i)  nearest neighbour distances of query points, and the kNN-CDFs for the discrete tracers
    #   (ii) the fraction of query points with nearest neighbour distance less than the angular
    #        distance and smoothed field greater than the overdensity threshold

    if Verbose: 
        step_2_start_time = time.perf_counter()
        print('\ninitiating step 2 (looping the tracer-field cross-correlation computations over the multiple tracer realisations)...')

    #-----------------------------------------------------------------------------------------------

    n_reals = TracerPosVector.shape[0]
    p_gtr_k_veclist, p_gtr_dt_list, p_gtr_k_dt_veclist = [], [], []

    Interpolated_Smoothed_Field = {}
    Delta_Threshold = {}

    #------------------------------------------------------------------------------------------------
    for k_ind, k in enumerate(kList):
             
            p_gtr_k_veclist.append(np.zeros((n_reals, len(RBins[k_ind]))))
            p_gtr_dt_list.append(np.zeros(len(RBins[k_ind])))
            p_gtr_k_dt_veclist.append(np.zeros((n_reals, len(RBins[k_ind]))))

    #------------------------------------------------------------------------------------------------

    for realisation, TracerPos in enumerate(TracerPosVector):

        if Verbose:
            start_time_real = time.perf_counter()
            print(f'\n\n--------------  Realisation {realisation+1}/{n_reals}  --------------\n')

        #-------------------------------------------------------------------------------------------

        #Tracer calculations

        #Building the tree
        if Verbose: 
            start_time_tree = time.perf_counter()
            print('\nbuilding the kdTree for the discrete tracer set ...')

        xtree = scipy.spatial.cKDTree(TracerPos, boxsize=BoxSize)

        if Verbose: 
            print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_tree))

        #-------------------------------------------------------------------------------------------

        #Calculating the NN distances
        if Verbose: 
            start_time_NN = time.perf_counter()
            print('\ncomputing the tracer NN distances ...')
        dists, _ = xtree.query(QueryPos, k=max(kList))
        vol = dists[:, np.array(kList)-1]
        if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_NN))

        #-------------------------------------------------------------------------------------------
    
        #Calculating the auto kNN-CDFs
        if Verbose: 
            start_time_CDF = time.perf_counter()
            print('\ncomputing the tracer auto-CDFs P_{>=k} ...')
        p_gtr_k_list = calc_kNN_CDF(vol, RBins)
        for k_ind, k in enumerate(kList):
            p_gtr_k_veclist[k_ind][realisation] = p_gtr_k_list[k_ind]
        if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_CDF))

        #-------------------------------------------------------------------------------------------

        #Tracer-field calculations

        if Verbose: 
            start_time_tf_cross = time.perf_counter()
            print('\ncomputing the tracer-field cross-correlation ...\n')

        for k_ind, k in enumerate(kList):
             
            if Verbose: 
                if realisation==0:
                    print('\tComputing P_(>dt) and P_(>=k, >dt) for k = {} ...'.format(k))
                else:
                    print('\tComputing P_(>=k, >dt) for k = {} ...'.format(k))

            for i, ss in enumerate(RBins[k_ind]):

                ss_str = str(ss)

                #-----------------------------------------------------------------------------------

                # Interpolate the smoothed field at the query positions
                if ss_str not in Interpolated_Smoothed_Field:
                    Interpolated_Smoothed_Field[ss_str] = CIC_3D_Interp(QueryPos, SmoothedFieldDict[ss_str], BoxSize)

                interp_field = Interpolated_Smoothed_Field[ss_str]
            
                #-------------------------------------------------------------------------------------------------

                # Compute the overdensity threshold for the smoothed field
                if ss_str not in Delta_Threshold:
                    Delta_Threshold[ss_str] = np.percentile(Interpolated_Smoothed_Field[ss_str], FieldConstPercThreshold)
                   
                delta_star_ss = Delta_Threshold[ss_str]    

                #-----------------------------------------------------------------------------------

                #Compute the fraction of query points satisfying the joint condition
                ind_gtr_k_dt = np.where((vol[:, k_ind] < ss) & (interp_field > delta_star_ss))
                p_gtr_k_dt_veclist[k_ind][realisation, i] = len(ind_gtr_k_dt[0])/QueryPos.shape[0]

                #-----------------------------------------------------------------------------------

                #Compute the fraction of query points with smoothed field exceeding the overdensity threshold
                if realisation==0: 
                    ind_gtr_dt = np.where(interp_field > delta_star_ss)
                    p_gtr_dt_list[k_ind][i] = len(ind_gtr_dt[0])/QueryPos.shape[0]

        if Verbose: print('\n\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time_tf_cross))

        #-------------------------------------------------------------------------------------------

        if Verbose: 
            print('\ntime taken for realisation {}: {:.2e} s.'.format(realisation+1, time.perf_counter()-start_time_real))

    #-----------------------------------------------------------------------------------------------

    if Verbose: 
        print('\n\n--------------  all realisations done!  --------------\n')
        print('\n\ttime taken for step 2: {:.2e} s.'.format(time.perf_counter()-step_2_start_time))

    #-----------------------------------------------------------------------------------------------

    if Verbose: print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))
    
    if ReturnSmoothedDict: 
        return p_gtr_k_veclist, p_gtr_dt_list, p_gtr_k_dt_veclist, SmoothedFieldDict
    else: 
        return p_gtr_k_veclist, p_gtr_dt_list, p_gtr_k_dt_veclist

####################################################################################################

#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################