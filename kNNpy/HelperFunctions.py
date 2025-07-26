####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
from scipy import interpolate
import time
import pyfftw
import warnings
import smoothing_library as SL
import MAS_library as MASL

####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

####################################################################################################

def cdf_vol_knn(vol):
    r'''
    Returns interpolating functions for empirical CDFs of the given $k$-nearest neighbour distances.
    
    Parameters
    ----------
    vol : numpy float array of shape ``(n_query, n_kNN)``
        Sorted array of nearest neighbour distances, where 'n_query' is the number of query points and 'n_kNN' is the number of nearest neighbours queried.

    Returns
    -------
    cdf: list of function objects
        list of interpolated empirical CDF functions that can be evaluated at desired distance bins.
    '''
    
    #-----------------------------------------------------------------------------------------------

    #Initialising a list to contain the interpolating functions
    cdf = []

    #-----------------------------------------------------------------------------------------------

    #Inferring the number of query points and nearest neighbours
    n = vol.shape[0]
    l = vol.shape[1]

    #-----------------------------------------------------------------------------------------------

    #Calculating the empirical CDF
    gof = ((np.arange(0, n) + 1) / (n*1.0))
    for c in range(l):
        ind = np.argsort(vol[:, c])
        s_vol= vol[ind, c]
        #Calculating the interpolating function
        cdf.append(interpolate.interp1d(s_vol, gof, kind='linear', bounds_error=False))
        
    return cdf

####################################################################################################

def calc_kNN_CDF(vol, bins):
    r'''
    Returns the kNN-CDFs for the given nearest-neighbour distances, evaluated at the given distance bins.

    Parameters
    ----------
    vol : numpy float array of shape ``(n_query, n_kNN)``
        2D array containing sorted 1D arrays of nearest-neighbour distances, where 'n_query' is the number of query points and 'n_kNN' is the number of nearest-neighbours queried. `vol[:, i]` should be the array with the sorted $k_i^{th}$ nearest-neighbour distances.
    bins : list of numpy float array
        list of distance scale arrays at which the CDFs need to be evaluated (units must be same as in `vol`).

    Returns
    -------
    data : list of numpy float array
        kNN-CDFs evaluated at the desired distance bins. ``data[i]`` is the $k_i$NN-CDF if ``vol[:, i]`` containts the $k_i^{th}$ nearest-neigbour distances.
    '''

    #-----------------------------------------------------------------------------------------------

    #Initialising the list of kNN-CDFs
    data = []

    #-----------------------------------------------------------------------------------------------

    #Computing the interpolated empirical CDFs using the nearest-neighbour distances
    cdfs = cdf_vol_knn(vol)

    #-----------------------------------------------------------------------------------------------

    #Looping over the nearest-neighbour indices
    for i in range(vol.shape[1]):

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

def create_query_3D(query_type, query_grid, BoxSize):
    '''
    Generates an array of query points; can be either randomly drawn from a uniform distribution defined over the box or put on a uniform grid.

    Parameters
    ----------
    query_type : {'grid', 'random'}, str
        the type of query points to be generated; should be 'grid' for query points defined on a uniform grid and 'random' for query points drawn from a uniform random distribution.
    query_grid : int
        the 1D size of the query points array; the total number of query points generated will be ``query_grid**3``.
    BoxSize : float
        the size of the 3D box of the input density field, in Mpc/h. Must be a positive real number, and must not be ``np.inf`` or ``np.nan``.

    Returns
    -------
    query_pos : numpy float array of shape ``(query_grid**3, 3)``
        array of query point positions. For each query point in the array, the first, second and third entries are the x, y and z coordinates respectively, in Mpc/h.

    Raises
    ------
    ValueError
        if `BoxSize` is not a positive real number less than infinity.
    ValueError
        if an unknown query type is provided.
        
    See Also
    --------
    kNNpy.HelperFunctions.create_query_2DA : generates query points in 2D angular coordinates.
    '''

    #Validating inputs

    if np.isnan(BoxSize) or np.isinf(BoxSize) or BoxSize<=0.0:
        raise ValueError(f"Invalid box size: {BoxSize}; please provide a positive real number less than infinity!")

    #Creating the query points

    if query_type == 'grid':

        #Creating a grid of query points
        x_ = np.linspace(0., BoxSize, query_grid, endpoint=False)
        y_ = np.linspace(0., BoxSize, query_grid, endpoint=False)
        z_ = np.linspace(0., BoxSize, query_grid, endpoint=False)

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
    
####################################################################################################

def smoothing_3D(field, Filter, grid, BoxSize, R=None, kmin=None, kmax=None, thickness=None, Verbose=False):
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
        if set to True, the time taken to complete each step of the calculation will be printed, by default False.

    Returns
    -------
    smoothed_field : numpy float array of shape field.shape
        the smoothed field.

    Raises
    ------
    ValueError
        If required parameters (like R, kmin, kmax, or thickness) are missing for the specified filter type.
    ValueError
        If the input field dimensions do not form a cubical box.
    ValueError
        If the grid size does not match the field dimensions.
    ValueError
        If an unknown filter name is provided.

    Notes
    -----
    - For real-space filters ('Top-Hat', 'Gaussian', 'Shell'), the radial scale R must be specified.
    - For the 'Shell' filter, thickness must also be specified.
    - For the 'Top-Hat-k' filter in Fourier space, kmin and kmax must be specified, while R and thickness are ignored.
    - Any unused parameters will trigger warnings but not stop execution.
    '''
    
    #-----------------------------------------------------------------------------------------------
    
    if Verbose:
        total_start_time = time.perf_counter()
        print("\nStarting smoothing ...")
        
    #-----------------------------------------------------------------------------------------------
    
    if not (field.shape[0] == field.shape[1] == field.shape[2]):
        raise ValueError("The box provided is not cubical.")
    elif field.shape[0] != grid:
        raise ValueError("Grid size provided does not match with dimensions of the cubical box.")
        
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
        print(f'Total time taken: {time.perf_counter() - total_start_time:.2e} s.')
        
    #-----------------------------------------------------------------------------------------------
    
    return smoothed_field

####################################################################################################

def create_smoothed_field_dict_3D(field, Filter, grid, BoxSize, bins, thickness=None, Verbose=False):
    r'''
    Creates a dictionary containing the continuous field smoothed at various radial distance scales.

    Parameters
    ----------
    field : numpy float array
        the 3D array of the continuous field that needs to be smoothed. 
    Filter : string
        the filter to be used for smoothing. Valid filter types are: 'Top-Hat', 'Gaussian', 'Shell'. 
    grid : int
        the grid size of the input density field, which should be field.shape[0] assuming a cubical box.
    BoxSize : float
        the size of the 3D box of the input density field, in Mpc/h.
    bins : list of numpy float array
        list of distances for each nearest neighbour. The $i^{th}$ element of the list should contain a numpy array of the desired distance scales for the $k_i^{th}$ nearest neighbour.
    thickness : float, optional
        the thickness of the shell used for smoothing. Only use this parameter when 'Shell' filter is used. The smoothing is done using a shell with inner radius R-thickness/2 and outer radius R+thickness/2.
    Verbose : bool, optional
        if set to True, the time taken to complete each step of the calculation will be printed, by default False.

    Returns
    -------
    SmoothedFieldDict : dict
        dictionary containing the continuous field smoothed at various radial distance scales. For example, `SmoothedFieldDict['50.0']`  represents the continuous map smoothed at a scale of 50 Mpc/h.

    Raises
    ------
    ValueError
        If required parameters (like bins or thickness) are missing for the specified filter type.
    ValueError
        If the input field dimensions do not form a cubical box.
    ValueError
        If the grid size does not match the field dimensions.
    ValueError
        If an unknown filter name is provided.

    Notes
    -----
    - This function only works for the real space filters, so 'Top-Hat-k' is not a valid filter for this function.
    - For the 'Shell' filter, thickness must be specified.
    - Any unused parameters will trigger warnings but not stop execution.
    '''
    
    #-----------------------------------------------------------------------------------------------
    
    # This function is only for smoothing in real space
    
    if Filter == 'Top-Hat-k':
        raise ValueError(f"Unknown filter: {Filter}")
        
    kmin = None
    kmax = None
    
    #-----------------------------------------------------------------------------------------------
    
    if Verbose: 
        total_start = time.perf_counter()
        print(f'\nSmoothing the density field over the radial distance scales...\n')

    #-----------------------------------------------------------------------------------------------

    #Initializing the dictionary
    SmoothedFieldDict = {}

    #-----------------------------------------------------------------------------------------------
    
    #Looping over the nearest neighbour indices as inferred from the length of 'bins'
    for i in range(len(bins)):

        #-------------------------------------------------------------------------------------------
        
        if Verbose: start = time.perf_counter()

        #-------------------------------------------------------------------------------------------
        
        for j, R in enumerate(bins[i]):
            SmoothedFieldDict[str(R)] = \
            smoothing_3D(field, Filter, grid, BoxSize, R, kmin, kmax, thickness, Verbose)

    #-----------------------------------------------------------------------------------------------
    
    if Verbose: print('\nTotal time taken for all scales: {:.2e} s.'.format(time.perf_counter()-total_start))

    return SmoothedFieldDict

####################################################################################################

def CIC_3D_Interp(pos, field, Boxsize):
    r'''
    Interpolates a 3D field onto particle positions using Cloud-In-Cell (CIC) interpolation.

    Parameters
    ----------
    field : numpy.ndarray of shape ``(Ng, Ng, Ng)``
        The 3D scalar field defined on a cubic grid with resolution 'Ng^3'.

    pos : numpy.ndarray of shape ``(Np, 3)``
        The positions of 'Np' particles. The columns represent x, y, and z coordinates. Units in Mpc/h
    
    Boxsize: float
            The side length of the cubic volume in the same units as `pos`.
    Returns
    -------
    fieldI : numpy.ndarray of shape ``(Np,)``
        The interpolated field values at the given particle positions.
    '''
    
    #-----------------------------------------------------------------------------------------------

    # define the array containing the value of the density field at positions pos
    density_interpolated = np.zeros(pos.shape[0], dtype=np.float32)

    #-----------------------------------------------------------------------------------------------

    # find the value of the density field at the positions pos
    MASL.CIC_interp(field, Boxsize, pos, density_interpolated)

    #-----------------------------------------------------------------------------------------------

    return density_interpolated

####################################################################################################

def kNN_excess_cross_corr(auto_cdf_list_1, auto_cdf_list_2, joint_cdf_list, k1_k2_list=None):
    r'''
    Computes the excess spatial cross-correlation (Banerjee & Abel 2023)[^1] between two tracers (discrete or continuous) from their joint kNN distributions (`joint_cdf_list`) and their respective kNN-CDFs (`auto_cdf_list_1`, `auto_cdf_list_2`).

    Parameters
    ----------
    auto_cdf_list_1 : list of numpy float array
        auto kNN-CDFs of the first set of tracers. If `k1_k2_list` is not ``None``, The $i^{th}$ element should be the $k_1^i$NN-CDF if the $i^{th}$ element of `k1_k2_list` is ($k_1^i$, $k_2^i$).
    auto_cdf_list_2 : list of numpy float array
        auto kNN-CDFs of the second set of tracers. If `k1_k2_list` is not ``None``, The $i^{th}$ element should be the $k_2^i$NN-CDF if where the $i^{th}$ element of `k1_k2_list` is ($k_1^i$, $k_2^i$).
    joint_cdf_list : list of numpy float array
        joint kNN distributions of the two tracer sets. If `k1_k2_list` is not ``None``, The $i^{th}$ element should be the joint {$k_1^i$, $k_2^i$}NN-CDF, where the $i^{th}$ element of `k1_k2_list` is ($k_1^i$, $k_2^i$).
        
    k1_k2_list : list of int tuples
        describes the kind of cross-correlations being computed (see notes for more details), by default `None`. Should be not None only if dealing with tracer-tracer cross-correlations

    Returns
    -------
    psi_list : list of numpy float array
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

    Note that the inputs must be self-consistent, which means the following must be ``True``

        len(joint_cdf_list)==len(auto_cdf_list_1) and len(joint_cdf_list)==len(auto_cdf_list_2) and len(joint_cdf_list)==len(k1_k2_list)
        
    For example, if

        k1_k2_list = [(1,1), (1,2)]

    then
        
        len(auto_cdf_list_1) == 2 and len(auto_cdf_list_2) == 2 and len(joint_cdf_list) == 2

    must hold, and the first (second) element of `joint_cdf_list` should be the joint {1,1}NN-CDF ({1,2}NN-CDF).
        
    If `None` is passed for tracer-tracer cross-correlations, the correlations are assumed to be between the same NN indices (eg. {1,1}NN-CDF, {2,2}NN-CDF).

    References
    ----------
    [^1]: Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour   distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stac3813), Volume 519, Issue 4, March 2023, Pages 4856–4868
    '''

    #-------------------------------------------------------------------------------------------

    psi_list = []

    #-------------------------------------------------------------------------------------------
    
    #Check for consistency:
    if k1_k2_list:
        if len(joint_cdf_list)!=len(k1_k2_list) or len(joint_cdf_list)!=len(auto_cdf_list_1) or len(joint_cdf_list)!=len(auto_cdf_list_2): 
            raise ValueError('Inconsistent input shapes')
    else:
        if len(joint_cdf_list)!=len(auto_cdf_list_1) or len(joint_cdf_list)!=len(auto_cdf_list_2): 
            raise ValueError('Inconsistent input: shapes not compatible with each other')
    
    for k in range(len(joint_cdf_list)):
        psi_list.append(joint_cdf_list[k]/(auto_cdf_list_1[k]*auto_cdf_list_2[k]))

    return psi_list
        
####################################################################################################

#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################
