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

def cdf_vol_knn(vol):

    '''
    Returns interpolating functions for emperical CDFs of the given k-nearest neighbour distances.

    #-----------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    
    vol: float array of shape (n, l) where 'n' is the number of query points and 'l' is the number
         of nearest neighbours queried
        Sorted array of nearest neighbour distances
        
    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    cdf: list
        list of interpolated emperical CDF functions that can be evaluated at desired distance bins
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

#---------------------------------------------------------------------------------------------------

def calc_kNN_CDF(vol, kMax, bins):

    '''
    Returns the kNN-CDFs for the given nearest-neighbour distances, evaluated at the given distance
    bins.
    
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

#---------------------------------------------------------------------------------------------------

def bl_th(l, ss):
    
    '''
    Computes Legendre expansion coefficients for the top-hat window function in angular coordinates
    
    Parameters
    ----------
    
    l: int array
        array of multipole numbers.

    ss: float
        angular scale (in radians) at which the field is to be smoothed.

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

#---------------------------------------------------------------------------------------------------

def top_hat_smoothing_2DA(skymap, scale, Verbose=False):

    '''
    Smooths the given map at the given scale using the top hat window function in harmonic space
    
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
        Defaults to 'False'

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

#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################
