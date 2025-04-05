####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
import scipy
from scipy import interpolate
import healpy as hp
import time
import copy

####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

####################################################################################################

def cdf_vol_knn(vol):

    '''
    Returns interpolating functions for emperical CDFs of the given k-nearest neighbour distances.

    #-----------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    
    vol: float array of shape (n, l) where 'n' is the number of query points and 'l' is the number
         of nearest neighbours queried
        Sorted array of nearest neighbour distances.
        
    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    cdf: list
        list of interpolated emperical CDF functions that can be evaluated at desired distance bins.
    '''
    
    #-----------------------------------------------------------------------------------------------

    #Initialising a list to contain the interpolating functions
    cdf = []

    #-----------------------------------------------------------------------------------------------

    #Inferring the number of query points and nearest neighbours
    n = vol.shape[0]
    l = vol.shape[1]

    #-----------------------------------------------------------------------------------------------

    #Calculating the emperical CDF
    gof = ((np.arange(0, n) + 1) / (n*1.0))
    for c in range(l):
        ind = np.argsort(vol[:, c])
        s_vol= vol[ind, c]
        #Calculating the interpolating function
        cdf.append(interpolate.interp1d(s_vol, gof, kind='linear', bounds_error=False))
        
    return cdf

####################################################################################################

def calc_kNN_CDF(vol, kMax, bins):

    '''
    Returns the kNN-CDFs for the given nearest-neighbour distances, evaluated at the given distance
    bins.

    #-----------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    
    vol: float array of shape (n, l) where 'n' is the number of query points and 'l' is the number
         of nearest neighbours queried
        Sorted array of nearest neighbour distances.
        
    kMax: int
        the number of nearest neighbours to calculate the distances to. For example, if kMax = 3, 
        the first 3 nearest-neighbour distributions will be computed.

    bins: list of float arrays
        list of distances for each nearest neighbour. The i^th element of the list should contain a
        numpy array of the desired distances for the i^th nearest neighbour.

    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    data: list of float arrays, each of shape len(bins[k-1]) for 1<=k<=kMax
        kNN-CDFs evaluated at the desired distance bins.
    '''

    #-----------------------------------------------------------------------------------------------
    
    #Initialising the list of kNN-CDFs
    data = []

    #-----------------------------------------------------------------------------------------------

    #Computing the interpolated emperical CDFs using the nearest-neighbour distances
    cdfs = cdf_vol_knn(vol)

    #-----------------------------------------------------------------------------------------------

    #Looping over the nearest-neighbour indices
    for i in range(kMax):

        #-------------------------------------------------------------------------------------------

        #Finding the minimum and maximum values of the NN distances
        min_dist = np.min(vol[:, i])
        max_dist = np.max(vol[:, i])

        #-------------------------------------------------------------------------------------------

        #Finding if any of the user-input bins lie outside the range spanned by the NN distances
        bin_mask = np.searchsorted(bins[i], [min_dist, max_dist])
        if bin_mask[1]!=len(bins[i]):
            if bins[i][bin_mask[1]] == max_dist:
                bin_mask[1] += 1

        #-------------------------------------------------------------------------------------------
                
        NNcdf = np.zeros(len(bins[i]))
        
        #Setting the value of the CDFs at scales smaller than the smallest NN distance to 0
        NNcdf[:bin_mask[0]] = 0
        
        NNcdf[bin_mask[0]:bin_mask[1]] = cdfs[i](bins[i][bin_mask[0]:bin_mask[1]])
        
        #Setting the value of the CDFs at scales larger than the largest NN distance to 1
        NNcdf[bin_mask[1]:] = 1
        
        data.append(NNcdf)
        
    return data

####################################################################################################

def bl_th(l, ss):
    
    '''
    Computes Legendre expansion coefficients for the top-hat window function in angular coordinates.

    #-----------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    
    l: int array
        array of multipole numbers.

    ss: float
        angular scale (in radians) at which the field is to be smoothed.

    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    num/den: float array of same size as input array 'l'
        array of Legendre expansion coefficients at each input multipole number.
    '''

    plm1 = scipy.special.eval_legendre(l-1, np.cos(ss))
    plp1 = scipy.special.eval_legendre(l+1, np.cos(ss))
    num = (plm1-plp1)
    den = (2*l+1)*(1-np.cos(ss))
    return num/den

####################################################################################################

def top_hat_smoothing_2DA(skymap, scale, Verbose=False):

    '''
    Smooths the given map at the given scale using the top hat window function in harmonic space.
    See Devaraju (2015) for a discussion and derivation of expressions used here.

    #-----------------------------------------------------------------------------------------------

    References:

        1. Devaraju (2015):
        Devaraju B., 2015, doctoralThesis, doi:10.18419/opus-3985, http://elib.uni-stuttgart.de/handle/11682/4002

    #-----------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    
    skymap: float array
        the healpy map of the continuous field that needs to be smoothed.
        The values of the masked pixels, if any, should be set to hp.UNSEEN.

    scale: float
        angular scale (in radians) at which the field is to be smoothed.
        Please ensure scale is between 0 and 2*pi.

    Verbose: Binary
        if set to True, the time taken to complete each step of the calculation will be printed.
        Defaults to False.

    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    smoothed_map_masked: float array of same size as input array 'skymap'
        the smoothed healpy map, keeping the masked pixels of the original map masked.
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
            smoothed_alm[ind_lm] = map_alm[ind_lm]*bl[l] 
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

def create_query(NSIDE_query, mask, tolerance):

    '''
    Computes the usable query points for the given mask (ie., query points at least a user-defined 
    threshold distance away from the mask edge) and returns the same, along with a HEALPix 
    'query mask' that has the following values:
    
        0: pixels outside the observational footprint
        1: pixels inside the footprint but too close to the mask edge (not usable)
        2: usable pixels

    See Gupta & Banerjee (2024) for a detailed discussion. Currently supports only query grids of the same
    size as the HEALpix grid on which the continuous overdensity field skymap is defined.

    #-----------------------------------------------------------------------------------------------

    References:

        1. Gupta & Banerjee (2024):
        Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources 
        with k-nearest neighbour distributions, Monthly Notices of the Royal Astronomical Society, 
        Volume 531, Issue 4, July 2024, Pages 4619–4639, https://doi.org/10.1093/mnras/stae1424
    
    #-----------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    
    NSIDE_query: int
        the HEALPix NSIDE of the query grid (needs to be the same as that of the continuous field
        and the mask). Must be a power of 2 (eg. 128, 256, 512, etc.)

    mask: int array of shape (12*NSIDE_query**2,)
        array with 0 and 1 indicating that the corresponding HEALPixel is outside and inside the 
        observational footprint, respectively.

    tolerance: float
        the minimum angular distance (in radians) a query point needs to be away from the mask edge
        to be considered usable.

    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    query_mask: float array of same size as input array 'mask'
        the HEALPix query mask.

    QueryPositions: float array of shape (N_usable_pix, 3)
        array of usable query point positions ('N_usable_pix' is the number of pixels that are
        sufficiently far away from the mask edge).
    '''

    #-----------------------------------------------------------------------------------------------

    #Check all inputs are consistent with the function requirement

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

    #-----------------------------------------------------------------------------------------------

    #Query points are defined on a Healpix grid, points in or close to the mask are removed

    #Getting number of pixels from the NSIDE for the query points
    NPIX = hp.nside2npix(NSIDE_query)    

    #-----------------------------------------------------------------------------------------------
    
    #Getting the query mask

    #2 means that the query point is usable
    query_mask = 2*np.ones(NPIX)

    pixels = np.arange(NPIX)
    qpos_arr = np.array(hp.pix2vec(NSIDE_query, pixels))
    #Looping over the initial query pixels
    for pix_ind, qpos in enumerate(qpos_arr.T):
        sel_pix_close = hp.query_disc(NSIDE_query, qpos, tolerance)
        #1 means that the query point is close to the mask edge
        if np.any(mask[sel_pix_close]==hp.UNSEEN): query_mask[pix_ind] = 1

    pix_inside_arr = hp.vec2pix(NSIDE_query, qpos_arr[0], qpos_arr[1], qpos_arr[2])
    #0 means that the query point is outside the footprint
    query_mask[mask[pix_inside_arr]==hp.UNSEEN] = 0

    #-----------------------------------------------------------------------------------------------

    #Getting the query positions

    query_pixels = np.arange(NPIX)[query_mask==2]
    
    #Getting the latitudes and longitudes in degrees
    QueryPositions_Deg = np.transpose(hp.pixelfunc.pix2ang(NSIDE_query, query_pixels, lonlat=True))
    QueryPositions_Deg[:, [0, 1]] = QueryPositions_Deg[:, [1, 0]]
    
    #converting to radians
    QueryPositions = np.deg2rad(QueryPositions_Deg)

    return query_mask, QueryPositions

####################################################################################################

def create_smoothed_field_dict(skymap, bins, query_mask, Verbose=False):

    '''
    Creates a dictionary containing the continuous field smoothed at various angular distance
    scales.

    #-----------------------------------------------------------------------------------------------

    References:

        1. Gupta & Banerjee (2024):
        Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources 
        with k-nearest neighbour distributions, Monthly Notices of the Royal Astronomical Society, 
        Volume 531, Issue 4, July 2024, Pages 4619–4639, https://doi.org/10.1093/mnras/stae1424
    
    #-----------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    
    skymap: float array
        the healpy map of the continuous field that needs to be smoothed.
        The values of the masked pixels, if any, should be set to hp.UNSEEN.

    bins: list of float arrays
        list of distances for each nearest neighbour. The i^th element of the list should contain a
        numpy array of the desired distances for the i^th nearest neighbour.
    
    query_mask: int array of same size as 'skymap'
        array with 0, 1 and 2 indicating that the corresponding HEALPixel is outside the mask,
        too close to mask boundary and sufficiently far away from the boundary, respectively.
        Refer to function 'create_query' defined above for creating the query mask and see 
        Gupta and Banerjee (2024) for more details).

    Verbose: Binary
        if set to True, the time taken to complete each step of the calculation will be printed.
        Defaults to False.

    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    SmoothedFieldDict: dictionary
        dictionary containing the continuous field masked within the observational footprint and 
        smoothed at various angular distance scales. For example, SmoothedFieldDict['0.215'] 
        represents the continuous map smoothed at a scale of 0.215 radians.
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

def kNN_excess_cross_corr(auto_cdf_list_1, auto_cdf_list_2, joint_cdf_list, k1_k2_list=None):

    '''
    Computes the excess spatial cross-correlation (Banerjee & Abel 2023) between two tracers 
    (discrete or continuous) from their joint kNN distributions ('joint_cdf_list') and their 
    respective kNN-CDFs ('auto_cdf_list_1', 'auto_cdf_list_2').

    #-----------------------------------------------------------------------------------------------

    References:

        1. Banerjee & Abel (2023):
        Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour 
        distributions, Monthly Notices of the Royal Astronomical Society, Volume 519, Issue 4, 
        March 2023, Pages 4856–4868, https://doi.org/10.1093/mnras/stac3813

    #-----------------------------------------------------------------------------------------------

    Parameters
    ----------

    auto_cdf_list_1: list of float arrays
        auto kNN-CDFs of the first set of tracers evaluated.
        
    auto_cdf_list_2: list of float arrays
        auto kNN-CDFs of the second set of tracers evaluated.
    
    joint_cdf_list: list of float arrays
        joint kNN distributions of the two tracer sets

    k1_k2_list: list of int tuples
        describes the kind of cross-correlations being computed. Should be None for every scenario 
        other than tracer-tracer cross-correlation, in which case it should provide the combinations
        of NN indices in the list of joint CDFs. For example, if you wish to compute the excess 
        cross correlation for the joint {1,1}, {1,2} and {2,1}NN-CDFs, then set 
            
            k1_k2_list = [(1,1), (1,2), (2,1)]

        Note that the tuples should be consistent with the 'joint_cdf_list'. For example, if

            k1_k2_list = [(1,1), (1,2)]

        then
        
            joint_cdf_list = 

        must hold.
        Defaults to None. If None is passed for tracer-tracer cross-correlations, the correlations
        are assumed to be between the same NN indices (eg. {1,1}NN-CDF, {2,2}NN-CDF).

    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    psi_list: list of float arrays
        excess spatial cross-correlation between the two tracer sets
    '''

    #-------------------------------------------------------------------------------------------

    psi_list = []

    #-------------------------------------------------------------------------------------------
    
    if k1_k2_list:

        #Check for consistency:
        if len(joint_cdf_list)!=len(k1_k2_list): 
            raise ValueError('Inconsistent input: shape of "joint_cdf_list" not consistent with that of "k1_k2_list"')
        for k, (k1, k2) in enumerate(k1_k2_list):
            psi_list.append(joint_cdf_list[k]/(auto_cdf_list_1[k1]*auto_cdf_list_2[k2]))

    #-------------------------------------------------------------------------------------------
    
    else:
        #Check for consistency:
        if len(joint_cdf_list)!=len(auto_cdf_list_1) or len(joint_cdf_list)!=len(auto_cdf_list_2): 
            raise ValueError('Inconsistent input: shapes not compatible with each other')
        for k in range(len(joint_cdf_list)):
            psi_list.append(joint_cdf_list[k]/(auto_cdf_list_1[k]*auto_cdf_list_2[k]))

    return psi_list
        
####################################################################################################

#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################
