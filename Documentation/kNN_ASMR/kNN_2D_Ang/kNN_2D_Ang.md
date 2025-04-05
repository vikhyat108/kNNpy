Module kNN_ASMR.kNN_2D_Ang.kNN_2D_Ang
=====================================

Functions
---------

`TracerFieldCross2DA(kMax, BinsRad, MaskedQueryPosRad, MaskedTracerPosRad, SmoothedFieldDict, FieldConstPercThreshold, Verbose=False)`
:   Returns the probabilities \(P_{>k}\), \(P_{>dt}\) and \(P_{>k,>dt}\) for \(1 \leq k \leq\) `kMax`, that quantify the extent of the spatial cross-correlation between the given discrete tracer positions (`MaskedTracerPosRad`) and the given continuous overdensity field (`SmoothedFieldDict`). 
        
    1. \(P_{>k}(\theta)\): 
        the kNN-CDF of the discrete tracers, evaluated at angular distance scale \(\theta\)
                
    2. \(P_{>{\rm dt}}(\theta)\): 
        the probability of the overdensity field smoothed with a top-hat filter of angular size \(\theta\) exceeding the given constant percentile density threshold
                
    3. \(P_{>k, >{\rm dt}}(\theta)\):
        the joint probability of finding at least 'k' tracers within a spherical cap of radius \(\theta\) AND the overdensity field smoothed at angular scale \(\theta\) exceeding the given density threshold
                
    The excess cross-correlation (Banerjee & Abel 2023)[^1] can be computed trivially from the quatities (see the 'HelperFunctions' module for a method to do this):
        
    \[\psi_{k, {\rm dt}} = P_{>k, >{\rm dt}}/(P_{>k} \times P_{>{\rm dt}})\]
    
    Parameters
    ----------
    kMax : int
        the number of nearest neighbours to calculate the distances to. For example, if `kMax` = 3, the first 3 nearest-neighbour distributions will be computed.
    BinsRad : list of array_like
        list of angular distances (in radians) for each nearest neighbour. The \(i^{th}\) element of the list should contain a numpy array of the desired distances for the \(i^{th}\) nearest neighbour.
    MaskedQueryPosRad : array_like of shape (n_query, 2)
        array of sky locations for the query points. The sky locations must be on a grid. For each query point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure `-np.pi/2 <= declination <= pi/2` and `0 <= right ascension <= 2*np.pi`.
    MaskedTracerPosRad : array_like of shape (n_tracer, 2)
        array of sky locations for the discrete tracers. For each query point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure `-np.pi/2 <= declination <= pi/2` and `0 <= right ascension <= 2*np.pi`.
    SmoothedFieldDict : dict
        dictionary containing the continuous field masked within the observational footprint and smoothed at various angular distance scales. For example, `SmoothedFieldDict['0.215']` represents the continuous map smoothed at a scale of 0.215 radians.
    FieldConstPercThreshold : float
        the percentile value for the constant percentile threshold to be used for the continuous field. For example, `FieldConstPercThreshold` = 75.0 represents a 75th percentile threshold.
    Verbose : bool, optional
        if set to True, the time taken to complete each step of the calculation will be printed, by default False.
    
    Returns
    -------
    p_gtr_k_list: list of array_like, each of shape 
                  len(`BinsRad`[k-1]) for 1 <= k <= `kMax`
        auto kNN-CDFs of the discrete tracers evaluated at the desired distance bins.
        
    p_gtr_dt_list: list of array_like, each of shape 
                   len(`BinsRad`[k-1]) for 1 <= k <= `kMax`
        continuum version of auto kNN-CDFs for the continuous field evaluated at the desired 
        distance bins.
    
    p_gtr_k_dt_list: list of array_like, each of shape 
                     len(`BinsRad`[k-1]) for 1 <= k <= `kMax`
        joint tracer-field nearest neighbour distributions evaluated at the desired distance bins
    
    Raises
    ------
    ValueError
        if the given query points are not on a two-dimensional grid.
    ValueError
        if declination of any of the query points is not in [``-np.pi/2``, ``np.pi/2``].
    ValueError
        if right ascension of any of the query points is not in [``0``, ``2*np.pi``].
    ValueError
        if declination of any of the tracer points is not in [``-np.pi/2``, ``np.pi/2``].
    ValueError
        if right ascension of any of the tracer points is not in [``0``, ``2*np.pi``].
    ValueError
        if the given tracer points are not on a two-dimensional grid.
    ValueError
        if the shape of field smoothed at a particular scale does not match the shape of the query point array.
    
    Notes
    -----
    Measures the angular cross-correlation between a set of discrete tracers and a continuous overdensity field using the k-nearest neighbour (kNN) formalism as defined in Banerjee & Abel (2023)[^1] and Gupta & Banerjee (2024)[^2].
    
    The field must already be smoothed at the desired angular distance scales using a top-hat filter (see the 'HelperFunctions' module for help with smoothing). The smoothed fields need to be provided as a dictionary ('SmoothedFieldDict'), see below for further details.
    
    Currently, the algorithm requires a constant percentile overdensity threshold for the continuous field and query points to be defined on a HEALPix grid. Extentions to a constant mass threshold and poisson-sampled query points on the sky may be added in the future. 
    
    Data with associated observational footprints are supported, in which case, only tracer positions within the footprint should be provided and the field should be masked appropriately.Importantly, in this case, query points need to be within the footprint and appropriately padded from the edges of the footprint (see Gupta & Banerjee (2024)[^2] for a detailed discussion). If the footprints of the tracer set and the field are different, a combined mask representing the intersection of the two footprints should be used (see the 'HelperFunctions' module for help with masking and creating the modified query positions).
    
    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour   distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stac3813), Volume 519, Issue 4, March 2023, Pages 4856–4868
        
    [^2]: Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639