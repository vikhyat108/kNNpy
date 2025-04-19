####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
import scipy
from scipy import interpolate
import healpy as hp
import time
import copy
import pyfftw
import warnings
import smoothing_library as SL

#Test comment

####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

####################################################################################################

def cdf_vol_knn(vol):
    r'''
    Returns interpolating functions for emperical CDFs of the given $k$-nearest neighbour distances.
    
    Parameters
    ----------
    vol : numpy float array of shape ``(n_query, n_kNN)``
        Sorted array of nearest neighbour distances, where 'n_query' is the number of query points and 'n_kNN' is the number of nearest neighbours queried.

    Returns
    -------
    cdf: list of function objects
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

def calc_kNN_CDF(vol, kList, bins):
    r'''
    Returns the kNN-CDFs for the given nearest-neighbour distances, evaluated at the given distance bins.

    Parameters
    ----------
    vol : numpy float array of shape ``(n_query, n_kNN)``
        Sorted array of nearest neighbour distances, where 'n_query' is the number of query points and 'n_kNN' is the number of nearest neighbours queried.
    kList : list of int
        the nearest neighbours for which the distances have been calculated to. For example, if `kList = [2, 4, 8]`, then `vol` should contain the sorted distances to the $2^{nd}$, $4^{th}$ and $8^{th}$ nearest-neighbours.
    bins : list of numpy float arrays
        list of distances for each nearest neighbour. The $i^{th}$ element of the list should contain a numpy array of the desired distance scales for the $i^{th}$ nearest neighbour.

    Returns
    -------
    data : list of numpy float arrays
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
    for i, k in enumerate(kList):

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

def create_query_2DA(NSIDE_query, mask, tolerance):
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
        the HEALPix NSIDE of the query grid (needs to be the same as that of the continuous field and the mask). Must be a power of 2 (eg. 128, 256, 512, etc.)
    mask : numpy int array of shape ``(12*NSIDE_query**2, )``
        array with 0 and 1 indicating that the corresponding HEALPixel is outside and inside the observational footprint, respectively.
    tolerance : float
        the minimum angular distance (in radians) a query point needs to be away from the mask edge
        to be considered usable.

    Returns
    -------
    query_mask : numpy float array of shape ``mask.shape``
        the HEALPix query mask.

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

    Notes
    -----
    Please refer to Gupta & Banerjee (2024)[^1] for a detailed discussion on creation of query point in presence of observational footprints that do not cover the full sky. The algorithm currently supports only query grids of the same size as the HEALpix grid on which the continuous overdensity field skymap is defined.

    References
    ----------
    [^1]: Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639
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

def create_query_3D(query_type, query_grid, BoxSize):
    '''
    Generates an array of query points; can be either randomly drawn from a uniform distribution defined over the box or put on a uniform grid.

    Parameters
    ----------
    query_type : str
        the type of query points to be generated; should be 'grid' for query points defined on a uniform grid and 'random' for query points drawn from a uniform random distribution.
    query_grid : int
        the 1D size of the query points array; the total number of query points generated will be ``query_grid**3``.
    BoxSize : float
        the size of the 3D box of the input density field, in Mpc/h.

    Returns
    -------
    query_pos : numpy float array of shape ``(query_grid**3, 3)``
        array of query point positions. For each query point in the array, the first, second and third entries are the x, y and z coordinates respectively, in Mpc/h.

    Raises
    ------
    ValueError
        if an unknown query type is provided.
    '''

    if query_type == 'grid':

        #Creating a grid of query points
        x_ = np.linspace(0., BoxSize, query_grid)
        y_ = np.linspace(0., BoxSize, query_grid)
        z_ = np.linspace(0., BoxSize, query_grid)

        x, y, z = np.array(np.meshgrid(x_, y_, z_, indexing='xy'))

        query_pos = np.zeros((query_grid**3, 3))
        query_pos[:, 0] = np.reshape(x, query_grid**3)
        query_pos[:, 1] = np.reshape(y, query_grid**3)
        query_pos[:, 2] = np.reshape(z, query_grid**3)

    elif query_type == 'random':

        #Creating a set of randomly distributed query points
        query_pos = np.random.rand(query_grid**3, 3)*BoxSize

    else:   
        raise ValueError(f"Unknown query type: {query_type}; please provide a valid query type")
    
    return query_pos

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
        the healpy map of the continuous field that needs to be smoothed. The values of the masked pixels, if any, should be set to `hp.UNSEEN`.
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
        the healpy map of the continuous field that needs to be smoothed. The values of the masked pixels, if any, should be set to `hp.UNSEEN`.
    bins : list of numpy float arrays
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
    `query_mask` is a numpy int array with 0, 1 and 2 indicating that the corresponding HEALPixel is outside the mask, too close to mask boundary and sufficiently far away from the boundary, respectively.Please Refer to the helper function method `create_query_2DA` for creating the query mask. See also Gupta and Banerjee (2024)[^1] for a discussion.

    References
    ----------
    [^1] Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae1424), Volume 531, Issue 4, July 2024, Pages 4619–4639
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

def smoothing_3d(field, Filter, grid, BoxSize, R=None, kmin=None, kmax=None, thickness=None, Verbose=False):
    r'''
    Smooths the given map at the given scale using a window function of choice in real or k-space. 
    
    Parameters
    ----------
    field : numpy float array
        the 3D array of the continuous field that needs to be smoothed. 
    Filter : string
        the filter to be used for smoothing. 'Top-Hat', 'Gaussian', 'Shell' are for real space, and 'Top-Hat-k' is a top-hat filter in k-space.
    grid : int
        the grid size of the input density field, which should be field.shape[0] assuming a cubical box.
    BoxSize : float
        the size of the 3D box of the input density field, in Mpc/h.
    R : float, optional
        radial scale (in Mpc/h) at which the field is to be smoothed. Only use this parameter for real space smoothing.
    kmin : float, optional
        the minimum value of the wavenumber. Only use this parameter when 'Top-Hat-k' filter is used.
    kmax : float, optional
        the maximum value of the wavenumber. Only use this parameter when 'Top-Hat-k' filter is used.
    thickness : float, optional
        the thickness of the shell used for smoothing. Only use this parameter when 'Shell' filter is used. The smoothing is done using a shell with inner radius R-thickness/2 and outer radius R+thickness/2.
    Verbose : bool, optional
        if set to `True`, the time taken to complete each step of the calculation will be printed, by default `False`.

    Returns
    -------
    smoothed_field : numpy float array of shape ``field.shape``
        the smoothed field.

    Raises
    ------
    ValueError
        If required parameters (like `R`, `kmin`, `kmax`, or `thickness`) are missing for the specified filter type.
    ValueError
        If an unknown filter name is provided.

    Notes
    -----
    - For real-space filters ('Top-Hat', 'Gaussian', 'Shell'), the radial scale `R` must be specified.
    - For the 'Shell' filter, `thickness` must also be specified.
    - For the 'Top-Hat-k' filter in Fourier space, `kmin` and `kmax` must be specified, while `R` and `thickness` are ignored.
    - Any unused parameters will trigger warnings but not stop execution.
    '''
    
    #-----------------------------------------------------------------------------------------------
    if Verbose:
        total_start_time = time.perf_counter()
        print("\nStarting smoothing ...")
    #-----------------------------------------------------------------------------------------------

    if Filter in ['Top-Hat', 'Gaussian']:
        if R is None:
            raise ValueError(f"R must be provided for {Filter} filter.")
        if kmin is not None or kmax is not None:
            warnings.warn("kmin and kmax are not used for real-space filters and will be ignored.")
        if thickness is not None:
            warnings.warn("thickness is not used for real-space filters and will be ignored.")
        
        W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads=1)
        field_k = pyfftw.interfaces.numpy_fft.rfftn(field)
        smoothed_field_k = field_k * W_k
        smoothed_field = pyfftw.interfaces.numpy_fft.irfftn(smoothed_field_k)
        
    #-----------------------------------------------------------------------------------------------

    elif Filter == 'Top-Hat-k':
        if kmin is None or kmax is None:
            raise ValueError("Both kmin and kmax must be provided for 'Top-Hat-k' filter.")
        if R is not None:
            warnings.warn("R is not used for 'Top-Hat-k' filter and will be ignored.")
        if thickness is not None:
            warnings.warn("thickness is not used for 'Top-Hat-k' filter and will be ignored.")
        
        R = 0.0
        W_k = SL.FT_filter(BoxSize, R, grid, Filter, kmin=kmin, kmax=kmax, threads=1)
        field_k = pyfftw.interfaces.numpy_fft.rfftn(field)
        smoothed_field_k = field_k * W_k
        smoothed_field = pyfftw.interfaces.numpy_fft.irfftn(smoothed_field_k)
        
    #-----------------------------------------------------------------------------------------------

    elif Filter == 'Shell':
        if R is None or thickness is None:
            raise ValueError("Both R and thickness must be provided for 'Shell' filter.")
        if kmin is not None or kmax is not None:
            warnings.warn("kmin and kmax are not used for 'Shell' filter and will be ignored.")
        
        if Verbose:
            print("\nGenerating shell-smoothed field ...")
        
        grid_cell_size = BoxSize / grid
        field_k = pyfftw.interfaces.numpy_fft.rfftn(field)

        x = np.fft.fftfreq(grid) * grid
        y = np.fft.fftfreq(grid) * grid
        z = np.fft.fftfreq(grid) * grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r_grid = grid_cell_size * np.sqrt(X**2 + Y**2 + Z**2)

        W = np.zeros((grid, grid, grid), dtype=np.float32)
        W[(r_grid >= R - thickness/2) & (r_grid <= R + thickness/2)] = 1.0
        W_k = pyfftw.interfaces.numpy_fft.rfftn(W)

        smoothed_field_k = field_k * W_k
        smoothed_field = pyfftw.interfaces.numpy_fft.irfftn(smoothed_field_k) / np.sum(W)

    #-----------------------------------------------------------------------------------------------

    else:
        raise ValueError(f"Unknown filter: {Filter}")
        
    #-----------------------------------------------------------------------------------------------

    if Verbose:
        print("Smoothing completed.")
        print('Total time taken: {:.2e} s.'.format(time.perf_counter() - total_start_time))
        
    #-----------------------------------------------------------------------------------------------

    return smoothed_field

####################################################################################################

def kNN_excess_cross_corr(auto_cdf_list_1, auto_cdf_list_2, joint_cdf_list, k1_k2_list=None):
    r'''
    Computes the excess spatial cross-correlation (Banerjee & Abel 2023)[^1] between two tracers (discrete or continuous) from their joint kNN distributions (`joint_cdf_list`) and their respective kNN-CDFs (`auto_cdf_list_1`, `auto_cdf_list_2`).

    Parameters
    ----------
    auto_cdf_list_1 : list of numpy float arrays
        auto kNN-CDFs of the first set of tracers.
    auto_cdf_list_2 : list of numpy float arrays
        auto kNN-CDFs of the second set of tracers.
    joint_cdf_list : list of numpy float arrays
        joint kNN distributions of the two tracer sets
    k1_k2_list : list of int tuples
        describes the kind of cross-correlations being computed (see notes for more details), by default `None`.

    Returns
    -------
    psi_list : list of numpy float arrays
        excess spatial cross-correlation between the two tracer sets.

    Raises
    ------
    ValueError
        if `k1_k2_list` is not `None` and `len(joint_cdf_list)!=len(k1_k2_list)`
    ValueError
        if `k1_k2_list` is `None` and `len(joint_cdf_list)!=len(auto_cdf_list_1) or len(joint_cdf_list)!=len(auto_cdf_list_2)`

    Notes
    -----
    The parameter `k1_k2_list` describes the kind of cross-correlations being computed. It should be set to `None` for every scenario other than tracer-tracer cross-correlation, in which case it should provide the combinations of NN indices for the two tracers sets being cross-correlated. 
    
    For example, if you wish to compute the excess cross correlation for the joint {1,1}, {1,2} and {2,1}NN-CDFs, then set
            
        k1_k2_list = [(1,1), (1,2), (2,1)]

    Note that the tuples should be consistent with the `joint_cdf_list`. For example, if

        k1_k2_list = [(1,1), (1,2)]

    then
        
        len(joint_cdf_list) == 2

    must hold, and the first (second) element of `joint_cdf_list` should be the joint {1,1}NN-CDF ({1,2}NN-CDF).
        
    If `None` is passed for tracer-tracer cross-correlations, the correlations are assumed to be between the same NN indices (eg. {1,1}NN-CDF, {2,2}NN-CDF), and the following must be `True`

        len(joint_cdf_list)==len(auto_cdf_list_1) and len(joint_cdf_list)==len(auto_cdf_list_2)

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour   distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stac3813), Volume 519, Issue 4, March 2023, Pages 4856–4868
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
