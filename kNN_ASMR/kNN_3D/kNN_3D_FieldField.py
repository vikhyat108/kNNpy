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


def FieldFieldCross3D(RBins, BoxSize, QueryPos, SmoothedFieldDict1, SmoothedFieldDict2, Field1ConstPercThreshold, Field2ConstPercThreshold, Verbose=False):
    r'''
    Returns the probabilities $P_{>{\rm dt1}}$ and $P_{\geq k,>{\rm dt2}}$ for the given thresholds of the respective fields, quantifying the extent of the spatial cross-correlation between two continuous overdensity fields in three-dimensional space.   

    1. $P_{>{\rm dt1}}(r)$: 
        The probability of the first overdensity field smoothed with a top-hat filter of radius $r$ exceeding the given constant percentile density threshold (`Field1ConstPercThreshold`).
    
    2. $P_{>{\rm dt2}}(r)$: 
        The probability of the second overdensity field smoothed with a top-hat filter of radius $r$ exceeding the given constant percentile density threshold (`Field2ConstPercThreshold`).
    
    3. $P_{>{\rm dt1}, >{\rm dt2}}(r)$:
        The joint probability of both overdensity fields smoothed at scale $r$ exceeding their respective density thresholds (as specified by the parameters `Field1ConstPercThreshold` and `Field2ConstPercThreshold`).
    
    The excess cross-correlation (Banerjee & Abel 2023)[^1] can be computed trivially from these quantities:
    
    $$\psi_{{\rm dt1}, {\rm dt2}} = \frac{P_{>{\rm dt1}, >{\rm dt2}}}{P_{>{\rm dt1}} \times P_{>{\rm dt2}}}$$

    Parameters
    ----------
    RBins : list of numpy float arrays
        List of radial distance arrays (in comoving Mpc/$h$) used for calculating the CDFs at different radial scales.

    BoxSize : float
        The size of the simulation box.

    QueryPos : numpy float array of shape ``(n_query, 3)``
        Array of 3D positions (e.g., in Cartesian or comoving coordinates) where the smoothed fields will be interpolated to compute the Joint CDF.

    SmoothedFieldDict1 : dict
        Dictionary containing the first continuous field smoothed at various radial scales. For example, ``SmoothedFieldDict1['3']`` represents the first field smoothed at a scale of 3 Mpc/$h$.

    SmoothedFieldDict2 : dict
        Dictionary containing the second continuous field smoothed at various radial scales. For example, ``SmoothedFieldDict2['3']`` represents the second field smoothed at a scale of 3 Mpc/$h$.

    Field1ConstPercThreshold : float
        The percentile threshold for identifying overdense regions in the first continuous field. For example, ``75.0`` indicates the 75th percentile.

    Field2ConstPercThreshold : float
        The percentile threshold for identifying overdense regions in the second continuous field. For example, ``75.0`` indicates the 75th percentile.

    Verbose : bool, optional
        If True, prints timing information for each step. Default is False.

    Returns
    -------
    p_gtr_dt1_list : list of numpy float arrays
        Overdensity-field1 auto kNN-CDFs evaluated at the given RBins radial scales.

    p_gtr_dt2_list : list of numpy float arrays
        Overdensity-field2 auto kNN-CDFs evaluated at the given RBins radial scales.

    p_gtr_dt1_dt2_list : list of numpy float arrays
        Joint CDFs of both overdensity fields exceeding their respective density thresholds at the given scales.

    Raises
    ------
    ValueError
        If input arrays are not 3D or have mismatched shapes.

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour distributions, [MNRAS](https://doi.org/10.1093/mnras/stac3813), Volume 519, Issue 4, March 2023, Pages 4856-4868

    [^2]: Eishica Chand, Arka Banerjee, Simon Foreman, Francisco Villaescusa-Navarro, [MNRAS](https://doi.org/10.1093/mnras/staf433), Volume 538, Issue 3, April 2025, Pages 2204-221 
    '''

    if Verbose: 
        total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
    # Step 0: Input validation
    if Verbose: 
        print('Checking inputs ...')

    if QueryPos.shape[1] != 3:
        raise ValueError("Query positions must be 3D (shape: n_query x 3).")
    
    if Verbose: 
        print('\tdone.')

    #-----------------------------------------------------------------------------------------------
    # Step 1: Compute the auto CDFs for the overdensity fields along with the joint CDFs

    if Verbose:
        step_1_start_time = time.perf_counter()
        print('\nInitiating step 1 ...')

    #------------------------------------------------------------------------------------------------

    # To store the CDFs for each k
    p_gtr_dt1     = np.zeros(len(RBins))
    p_gtr_dt2     = np.zeros(len(RBins))
    p_gtr_dt1_dt2 = np.zeros(len(RBins))

    #------------------------------------------------------------------------------------------------

    # Loop over the scales in the smoothed field dictionary
    for i, ss in enumerate(RBins):

        #------------------------------------------------------------------------------------------------

        # Load the smoothed field and interpolate it to the query positions
        interp_field1 = CIC_3D_Interp(SmoothedFieldDict1[str(ss)], BoxSize, QueryPos)
        interp_field2 = CIC_3D_Interp(SmoothedFieldDict2[str(ss)], BoxSize, QueryPos)

        # Check if the interpolated field has the same number of points as QueryPos
        if interp_field1.shape[0] != QueryPos.shape[0]:
            raise ValueError(f"Mismatch in shape of interpolated field1 for scale {ss:.3e}")
        
        if interp_field2.shape[0] != QueryPos.shape[0]:
            raise ValueError(f"Mismatch in shape of interpolated field2 for scale {ss:.3e}")
        
        # Compute the delta threshold/cutoff for the first field
        delta_cut1 = np.percentile(interp_field1, Field1ConstPercThreshold)
        delta_cut2 = np.percentile(interp_field2, Field2ConstPercThreshold)

        # Compute the fraction of the sphere's enclosing greater than delta threshold
        field_mask1    = interp_field1 > delta_cut1
        p_gtr_dt1[i]   = np.sum(field_mask1) / QueryPos.shape[0]

        field_mask2    = interp_field2 > delta_cut2
        p_gtr_dt2[i]   = np.sum(field_mask2) / QueryPos.shape[0]

        # Compute the joint probability of both fields exceeding their respective thresholds
        joint_mask     = field_mask1 & field_mask2
        p_gtr_dt1_dt2[i] = np.sum(joint_mask) / QueryPos.shape[0]

        #------------------------------------------------------------------------------------------------
    
    if Verbose:
        print('\t\tdone; time taken for step 1: {:.2e} s.'.format(time.perf_counter() - step_1_start_time))

    #-----------------------------------------------------------------------------------------------
    # Step 2: Return the results

    if Verbose:
        print(f"\nTotal time taken: {time.perf_counter() - total_start_time:.2e} s")

    return p_gtr_dt1, p_gtr_dt2, p_gtr_dt1_dt2