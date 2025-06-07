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
from kNN.ASMR.HelperFunctions import CIC_3D_Interp


def TracerFieldCross3D(kList, RBins, BoxSize, QueryPos, TracerPos, SmoothedFieldDict, FieldConstPercThreshold, Verbose=False):
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

    QueryPos : numpy float array of shape ``(n_query, 3)``
        Array of 3D positions (e.g., in Cartesian or comoving coordinates) used to query the nearest-neighbour distances, and also compute field's CDF.

    TracerPos : numpy float array of shape ``(n_tracer, 3)``
        Array of 3D positions of discrete tracers, with columns representing the x, y, and z coordinates, respectively.
    
    SmoothedFieldDict : dict
        dictionary containing the continuous fields (on a grid) smoothed at various radial scales. For example, ``SmoothedFieldDict['3']`` represents the continuous map smoothed at a scale of 3Mpc/h.

    FieldConstPercThreshold : float
        The percentile threshold for identifying overdense regions in the continuous field. For example, ``75.0`` indicates the 75th percentile.

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

    Raises
    ------
    ValueError
        If input arrays are not 3D or have mismatched shapes.

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

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------
    # Step 1: Compute kNN-CDFs for tracer positions
    if Verbose:
        step_1_start_time = time.perf_counter()
        print('\ninitiating step 1 ...')

    if Verbose:
        print('\n\tbuilding the kdTree ...')
        t_start = time.perf_counter()
    xtree = scipy.spatial.cKDTree(TracerPos, boxsize=BoxSize)

    if Verbose:
        print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter() - t_start))

    if Verbose:
        print('\n\tcomputing the tracer NN distances ...')
        t_start = time.perf_counter()
    dists, idx = xtree.query(QueryPos, k=max(kList), n_jobs=-1)
    del idx
    gc.collect()

    # Ensure dists is 2D (important if max(kList) == 1)
    if len(dists.shape) == 1:
        dists = dists[:, np.newaxis]
    vol = [dists[:, k-1] for k in kList]

    if Verbose:
        print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter() - t_start))

    if Verbose:
        print('\n\tcomputing P_{>=k} ...')
        t_start = time.perf_counter()
    p_gtr_k_list = calc_kNN_CDF(vol, RBins)

    if Verbose:
        print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter() - t_start))
        print('time taken for step 1: {:.2e} s.'.format(time.perf_counter() - step_1_start_time))

    #-----------------------------------------------------------------------------------------------
    # Step 2: Compute kNN-CDFs for the overdensity field, and the joint CDFs with tracers 
    
    if Verbose:
        step_2_start_time = time.perf_counter()
        print('\ninitiating step 2 ...')

    #------------------------------------------------------------------------------------------------

    # Precompute all interpolated fields, and the delta threshold for each scale
    Interpolated_Smoothed_Field = {}
    Delta_Threshold = {}

    #------------------------------------------------------------------------------------------------

    # Loop over the scales in the smoothed field dictionary
    for i, ss in enumerate(RBins):

        #------------------------------------------------------------------------------------------------

        # Load the smoothed field and interpolate it to the query positions
        SmoothedField = SmoothedFieldDict[str(ss)]
        interp_field = CIC_3D_Interp(SmoothedField, BoxSize, QueryPos)

        # Check if the interpolated field has the same number of points as QueryPos
        if interp_field.shape[0] != QueryPos.shape[0]:
            raise ValueError(f"Mismatch in shape of interpolated field for scale {ss:.3e}")

        # Store the interpolated field and delta threshold in the dictionaries
        Interpolated_Smoothed_Field[str(ss)] = interp_field
        Delta_Threshold[str(ss)] = np.percentile(interp_field, FieldConstPercThreshold)
    
    #------------------------------------------------------------------------------------------------

    # To store the CDFs for each k
    p_gtr_k_dt_list = []
    p_gtr_dt_list = []

    #------------------------------------------------------------------------------------------------

    # Compute the CDFs
    for k_ind, k in enumerate(kList):

        if Verbose:
            print(f"\nComputing P_{{>=k, >dt}} and P_{{>dt}} for k = {k} ...")
            k_start_time = time.perf_counter()

        p_gtr_k_dt = np.zeros(len(RBins[k]))
        p_gtr_dt   = np.zeros(len(RBins[k]))

        for j, ss in enumerate(RBins[k]):

            #------------------------------------------------------------------------------------------------
            # Load the interpolated field and delta threshold for the current scale
            interp_field = Interpolated_Smoothed_Field[str(ss)]
            delta_star   = Delta_Threshold[str(ss)]

            # Compute the fraction of sphere's enclosing k nearest neighbours
            vol_mask    = vol[:, k_ind] < ss
            p_gtr_dt[j] = np.sum(field_mask) / QueryPos.shape[0]


            # Compute the fraction of sphere's enclosing greater than delta threshold
            field_mask    = interp_field > delta_star
            p_gtr_k_dt[j] = np.sum(field_mask & vol_mask) / QueryPos.shape[0]

        #------------------------------------------------------------------------------------------------
        
        p_gtr_k_dt_list.append(p_gtr_k_dt)
        p_gtr_dt_list.append(p_gtr_dt)

        #------------------------------------------------------------------------------------------------

        if Verbose:
            print(f"\tdone for k = {k}; time taken: {time.perf_counter() - k_start_time:.2e} s")

        #------------------------------------------------------------------------------------------------

    if Verbose:
        print(f"\nTotal time taken: {time.perf_counter() - step_2_start_time:.2e} s")

    return p_gtr_k_dt_list, p_gtr_dt_list
