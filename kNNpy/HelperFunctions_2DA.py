####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
import scipy
import healpy as hp
import time
import copy

####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

####################################################################################################

def create_query_2DA(NSIDE_query, mask, tolerance, Verbose=False):
    r'''
    Computes the usable query points for the given mask (ie., query points at least a user-defined 
    threshold distance away from the mask edge) and returns the same, along with a HEALPix 
    `query mask` that has the following values:
    
        0: pixels outside the observational footprint
        1: pixels inside the footprint but too close to the mask edge (not usable)
        2: usable pixels

    Parameters
    ----------
    NSIDE_query : int
        the HEALPix NSIDE of the query grid (needs to be the same as that of the continuous field and the mask). Must be a power of 2 (eg. 128, 256, 512, etc.).
    mask : numpy float array of shape ``(12*NSIDE_query**2, )``
        array encoding the observational footprint associated with the data. The value of the mask should be ``1.0`` for HEALPixels inside the observational footprint and ``healpy.UNSEEN`` for HEALPixels outside the observational footprint. ``healpy.UNSEEN = -1.6375e+30`` is a special value for masked pixels used by the ``healpy`` package. If there is no observational footprint (for example, data such as gravitational wave catalogs that are all-sky, or simulated datasets), please enter an array with all values equal to ``1.0``.
    tolerance : float
        the minimum angular distance (in radians) a query point needs to be away from the mask edge
        to be considered usable.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    query_mask : numpy int array of shape ``mask.shape``
        the HEALPix query mask, i.e., an array with 0, 1 and 2 indicating that the corresponding HEALPixel is outside the observational footprint, too close to mask boundary and sufficiently inside the observational footprint (far from the boundary), respectively.

    QueryPositions : numpy float array of shape ``(N_usable_pix, 2)``
        array of usable query point positions, where 'N_usable_pix' is the number of pixels that are sufficiently far away from the mask edge, as determined by this method. For each query point in the array, the first (second) coordinate is the declination (right ascension) in radians.

    Raises
    ------
    ValueError
        if `tolerance` is not in `[0, 2*np.pi]`
    ValueError
        if `NSIDE_query` is not a power of 2
    ValueError
        if `NSIDE_query` is not the same as the NSIDE of the continuous field and the mask
        
    See Also
    --------
    kNN_ASMR.HelperFunctions.create_query_3D : generates query points in 3D.

    Notes
    -----
    Please refer to Gupta & Banerjee (2024)[^1] for a detailed discussion on creation of query point in presence of observational footprints that do not cover the full sky. The algorithm currently supports only query grids of the same size as the HEALpix grid on which the continuous overdensity field skymap is defined.

    References
    ----------
    [^1]: Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639
    '''

    #-----------------------------------------------------------------------------------------------

    #Check all inputs are consistent with the function requirement

    if Verbose: 
        total_start = time.perf_counter()
        print('\nValidating inputs...')

    #Check if 'tolerance' is a valid angle
    if tolerance<0 or tolerance>2*np.pi:
        raise ValueError('Invalid threshold distance: please ensure 0 <= tolerance <= 2*pi.')

    #Check if 'NSIDE_query' is a power of 2
    if not (NSIDE_query > 0 and (NSIDE_query & (NSIDE_query - 1)) == 0):
        raise ValueError('Invalid NSIDE for the query grid: please ensure NSIDE is a power of 2')

    #Getting the NSIDE for the original mask
    NSIDE_mask = hp.get_nside(mask)

    #Check if the NSIDEs match
    if NSIDE_mask!=NSIDE_query:
        raise ValueError(f'NSIDE of the query grid ({NSIDE_query}) does not match NSIDE of \
        the mask ({NSIDE_mask}).')
    
    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #Query points are defined on a Healpix grid, points in or close to the mask are removed

    if Verbose: print('\nDefining a query mask...')

    #Getting number of pixels from the NSIDE for the query points
    NPIX = hp.nside2npix(NSIDE_query)    

    #-----------------------------------------------------------------------------------------------
    
    #Getting the query mask

    #2 means that the query point is usable
    query_mask = 2*np.ones(NPIX)

    #Check if non-trivial observational mask is present
    if np.any(mask == hp.UNSEEN):

        if Verbose: 
            print('\tNon-trivial observational mask detected. Now removing the query points outside the observational footprint and the query points too close to the mask boundary...')

        pixels = np.arange(NPIX)
        qpos_arr = np.array(hp.pix2vec(NSIDE_query, pixels))
        #Looping over the initial query pixels
        for pix_ind, qpos in enumerate(qpos_arr.T):
            perc_progress = int((pix_ind+1)*100/len(qpos_arr.T))
            perc_progress_2 = int((pix_ind+2)*100/len(qpos_arr.T))
            if Verbose and perc_progress%10==0 and perc_progress!=perc_progress_2: print('\t{}% done'.format(perc_progress))
            sel_pix_close = hp.query_disc(NSIDE_query, qpos, tolerance)
            #1 means that the query point is close to the mask edge
            if np.any(mask[sel_pix_close]==hp.UNSEEN): query_mask[pix_ind] = 1

        pix_inside_arr = hp.vec2pix(NSIDE_query, qpos_arr[0], qpos_arr[1], qpos_arr[2])
        #0 means that the query point is outside the footprint
        query_mask[mask[pix_inside_arr]==hp.UNSEEN] = 0

    #-----------------------------------------------------------------------------------------------

    #Getting the query positions

    if Verbose: print('\nGetting the query positions...')

    query_pixels = np.arange(NPIX)[query_mask==2]
    
    #Getting the latitudes and longitudes in degrees
    QueryPositions_Deg = np.transpose(hp.pixelfunc.pix2ang(NSIDE_query, query_pixels, lonlat=True))
    QueryPositions_Deg[:, [0, 1]] = QueryPositions_Deg[:, [1, 0]]
    
    #converting to radians
    QueryPositions = np.deg2rad(QueryPositions_Deg)

    if Verbose:
        print('\tdone.')
        print('\nTotal time taken: {:.2e} s'.format(time.perf_counter()-total_start))

    return query_mask, QueryPositions

####################################################################################################

def bl_th(l, ss):
    r'''
    Computes Legendre expansion coefficients for the top-hat window function in angular coordinates.


    Parameters
    ----------
    l : numpy int array
        array of multipole numbers.
    ss : float
        angular scale (in radians) at which the field is to be smoothed.

    Returns
    -------
    numpy float array of shape ``l.shape``
        array of Legendre expansion coefficients at each input multipole number.
    '''

    plm1 = scipy.special.eval_legendre(l-1, np.cos(ss))
    plp1 = scipy.special.eval_legendre(l+1, np.cos(ss))
    num = (plm1-plp1)
    den = 4*np.pi*(1-np.cos(ss))
    return num/den

####################################################################################################

def top_hat_smoothing_2DA(skymap, scale, Verbose=False):
    r'''
    Smooths the given map at the given scale using the top hat window function in harmonic space.

    Parameters
    ----------
    skymap : numpy float array
        the healpy map of the continuous field that needs to be smoothed. The values of the masked pixels, if any, should be set to `healpy.UNSEEN`. ``healpy.UNSEEN = -1.6375e+30`` is a special value for masked pixels used by the ``healpy`` package.
    scale : float
        angular scale (in radians) at which the field is to be smoothed. Please ensure `scale` is between `0` and `2*np.pi`.
    Verbose : bool, optional
        if set to `True`, the time taken to complete each step of the calculation will be printed, by default `False`.

    Returns
    -------
    smoothed_map_masked : numpy float array of shape ``skymap.shape``
        the smoothed healpy map, keeping the masked pixels of the original map masked.

    Raises
    ------
    ValueError
        if `scale` is not in `[0, 2*np.pi]`
        
    See Also
    --------
    kNN_ASMR.HelperFunctions.smoothing_3d : performs smoothing operations in 3D.

    Notes
    -----
    The following expression is used to compute the the spherical harmonic expansion coefficients $\alpha^{\theta}_{\ell m}$ of the field smoothed at angular scale $\theta$ using a top hat window function (See Devaraju (2015)[^1] and Gupta & Banerjee (2024)[^2] for derivations and a detailed discussion)
    $$\alpha^{\theta}_{\ell m} = 4\pi\frac{b_{\ell}} {2\ell+1}\alpha_{\ell m},$$
    where $b_{\ell}$ are the the Legedre expansion coefficients of the top hat function, given by
    $$b_{\ell} = \frac{1}{4\pi(1-\cos\theta)}\left[P_{\ell-1}(\cos\theta)-P_{\ell+1}(\cos\theta)\right].$$
    The smoothed field is reconstructed from $\alpha^{\theta}_{\ell m}$ using healpy's `alm2map` method.


    References
    ----------
    [^1]: Devaraju B., 2015, [doctoralThesis](http://elib.uni-stuttgart.de/handle/11682/4002), doi:10.18419/opus-3985.
    [^2]: Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639
    '''

    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------

    #Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if scale<0 or scale>2*np.pi:
        raise ValueError('Invalid angular smoothing scale: please ensure 0 <= scale <= 2*pi.')

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------
    
    #Getting the NSIDE of the map
    NSIDE = hp.get_nside(skymap)

    #Using the default l_max for the harmonic expansion
    l_max = 3*NSIDE-1

    #-----------------------------------------------------------------------------------------------

    #Computing the spherical harmonic expansion for the map
    if Verbose: 
        print('\nComputing the spherical harmonic expansion for the map ...')
    l_arr = np.array(range(l_max+1))                                     
    map_alm = hp.map2alm(skymap, use_pixel_weights=False, lmax=l_max)
    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #Getting the legendre coefficients of the top hat window function
    if Verbose: 
        print('\nGetting the legendre coefficients of the top hat window function ...')
    bl = bl_th(l_arr, scale)
    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #Getting the spherical harmonic expansion of the smoothed map
    if Verbose: 
        print('\nGetting the spherical harmonic expansion of the smoothed map ...')
    smoothed_alm = np.zeros(map_alm.shape[0], dtype = 'complex')    
    for l in range(l_max+1):
        for m in range(l+1):
            ind_lm = hp.Alm.getidx(l_max, l, m)
            smoothed_alm[ind_lm] = map_alm[ind_lm]*bl[l]*4*np.pi/(2*l+1)
    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #Generating the smoothed map from the harmonic coeffients
    if Verbose: 
        print('\nGenerating the smoothed map from the harmonic coeffients ...')
    smoothed_map = hp.alm2map(smoothed_alm, nside=NSIDE)
    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #If a mask is present
    smoothed_map_masked = copy.deepcopy(smoothed_map)                    
    smoothed_map_masked[skymap==hp.UNSEEN] = hp.UNSEEN

    #-----------------------------------------------------------------------------------------------

    if Verbose: print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))

    return smoothed_map_masked

####################################################################################################

def create_smoothed_field_dict_2DA(skymap, bins, query_mask, Verbose=False):
    r'''
    Creates a dictionary containing the continuous field smoothed at various angular distance scales.

    Parameters
    ----------
    skymap : numpy float array
        the healpy map of the continuous field that needs to be smoothed. The values of the masked pixels, if any, should be set to `healpy.UNSEEN`. ``healpy.UNSEEN = -1.6375e+30`` is a special value for masked pixels used by the ``healpy`` package.
    bins : list of numpy float array
        list of distances for each nearest neighbour. The $i^{th}$ element of the list should contain a numpy array of the desired distance scales for the $i^{th}$ nearest neighbour.
    query_mask : numpy float array of shape ``skymap.shape``
        the HEALPix query mask.
    Verbose : bool, optional
        if set to `True`, the time taken to complete each step of the calculation will be printed, by default `False`.

    Returns
    -------
    SmoothedFieldDict : dict
        dictionary containing the continuous field masked within the observational footprint and smoothed at various angular distance scales. For example, `SmoothedFieldDict['0.215']`  represents the continuous map smoothed at a scale of 0.215 radians.

    Notes
    -----
    `query_mask` is a numpy int array with 0, 1 and 2 indicating that the corresponding HEALPixel is outside the mask, too close to mask boundary and sufficiently far away from the boundary, respectively. Please Refer to the helper function method `create_query_2DA()` for creating the query mask. See also Gupta and Banerjee (2024)[^1] for a discussion.

    References
    ----------
    [^1]: Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639
    '''

    #-----------------------------------------------------------------------------------------------
    
    if Verbose: 
        total_start = time.perf_counter()
        print(f'\nSmoothing the density field over the given angular distance scales...\n')

    #-----------------------------------------------------------------------------------------------

    #Initializing the dictionary
    SmoothedFieldDict = {}

    #-----------------------------------------------------------------------------------------------
    
    #Looping over the nearest neighbour indices as inferred from the length of 'bins'
    for i in range(len(bins)):

        #-------------------------------------------------------------------------------------------
        
        if Verbose: start = time.perf_counter()

        #-------------------------------------------------------------------------------------------
        
        for j, ss in enumerate(bins[i]):
            #Square bracket selects only those pixels that are not close to the mask boundaries
            #Turning verbose off to avoid too much text output
            SmoothedFieldDict[str(ss)] = \
            top_hat_smoothing_2DA(skymap, ss, Verbose=False)[query_mask==2]

        #-------------------------------------------------------------------------------------------
        
        if Verbose: 
            print('\tdistance scales for {}NN done; time taken: {:.2e} s.'.format(i+1, time.perf_counter()-start))

    #-----------------------------------------------------------------------------------------------
    
    if Verbose: print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start))

    return SmoothedFieldDict

####################################################################################################

#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################
