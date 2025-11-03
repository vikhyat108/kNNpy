####################################################################################################

#-------------------------------  Importing the required libraries  --------------------------------

import numpy as np
import time
import pyfftw
import MAS_library as MASL
import sys
import os

#Necessary for relative imports (see https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im)
module_path = os.path.abspath(os.path.join('../../'))
'''
@private
'''
if module_path not in sys.path:
    sys.path.append(module_path)

from kNNpy.HelperFunctions import create_smoothed_field_dict_3D

####################################################################################################

# Function that returns the the 2-point Cross-Correlation Function between a set of discrete tracers and a continuous field using the stacking method

def CrossCorr2pt(boxsize, bins, QueryPos, TracerPos, delta, thickness, R, kmin=None, kmax=None, Verbose=False):
    '''
    Calculates the Two-point Cross-correlation function between a set of tracers and a field. The interpolation can only be done using the 
    CIC-mass assignment scheme.

    Parameters
    ----------
    boxsize: float
        The length (in Mpc/h) of the cubic box containing the tracers and the field

    bins : float array of shape (m,)
        Set of m radial distances at which the 2PCF will be computed
        
    QueryPos : float array of shape (n_query,3) where n_query is the number of query points
            3D positions of the random query points inside the box, given in Cartesian coordinates (x, y, z) within the range         [0, boxsize]

    TracerPos : float array of shape (n_pos, 3) where n_pos is the number of discrete tracers
        3D positions of the n tracers (e.g., galaxies) inside the box, given in Cartesian coordinates (x, y, z) within the range         [0, boxsize]

    delta : float array of shape (ngrid, ngrid, ngrid)
        Smoothed overdensity field defined on a uniform grid with ngrid³ points

    thickness : float, optional
        the thickness of the shell used for smoothing. Only use this parameter when 'Shell' filter is used. The smoothing is done using a shell with inner radius R-thickness/2 and outer radius R+thickness/2.

    R : float, optional
        radial scale (in Mpc/h) at which the field is to be smoothed. Only use this parameter for real space smoothing.

    kmin : float, optional
        the minimum value of the wavenumber. Only use this parameter when 'Top-Hat-k' filter is used.

    kmax : float, optional
        the maximum value of the wavenumber. Only use this parameter when 'Top-Hat-k' filter is used.
    
    Verbose : bool, optional
        if set to True, the time taken to complete each step of the calculation will be printed, by default False.
    
    Returns
    -------
    xi : float array of shape (m,)
        The 2-point cross-correlation function (2PCF) between the tracer positions and the field, computed at each of the m radial bins.
    
    Raises
    ------
    ValueError
        if the given query points are not on a three-dimensional grid.
    ValueError
        if x,y, or z coordinate of any of the query points is not in ``(0, boxsize)``.
    ValueError
        if x,y, or z coordinate of any of the tracer points is not in ``(0, boxsize)``..
    ValueError
        if the given tracer points are not on a three-dimensional grid.
    ValueError
        if the given field is not a cubical three-dimensional array.
    '''
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    # Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if QueryPos.shape[1]!=3: 
        raise ValueError('Incorrect spatial dimension for query points: array containing the query point positions must be of shape (n_query, 3), ' \
        'where n_query is the number of query points.')

    if np.any(QueryPos[:, 0] <= 0 or QueryPos[:, 0] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < x < boxsize.')

    if np.any(QueryPos[:, 1] <= 0 or QueryPos[:, 1] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < y < boxsize.')

    if np.any(QueryPos[:, 2] <= 0 or QueryPos[:, 2] >= boxsize):
        raise ValueError('Invalid query point position(s): please ensure 0 < z < boxsize.')

    if np.any(TracerPos[:, 0] <= 0 or TracerPos[:, 0] >= boxsize):
        raise ValueError('Invalid tracer point position(s): please ensure 0 < x < boxsize.')

    if np.any(TracerPos[:, 1]<= 0 or TracerPos[:, 1]>= boxsize):
        raise ValueError('Invalid tracer point position(s): please ensure 0 < y < boxsize.')

    if np.any(TracerPos[:, 2]<= 0 or TracerPos[:, 2]>= boxsize):
        raise ValueError('Invalid tracer point position(s): please ensure 0 < z < boxsize.')

    if TracerPos.shape[1]!=3: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 3), ' \
        'where n_tracer is the number of tracers.')

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------
    
    # Calculating the number of grid points along each axis of the field delta
    shape = np.shape(delta)
    if len(shape) != 3 or shape[0] != shape[1] or shape[1] != shape[2]:
        raise ValueError("Error: Input array is not cubical (n, n, n).")
    ngrid = shape[0]

    # Calculating the grid cell size
    grid_cell_size = boxsize / ngrid  

    # Fourier Transforming the delta overdensity field
    pyfftw.interfaces.cache.enable()
    delta_k = pyfftw.interfaces.numpy_fft.rfftn(delta, threads=threads)

    # Initialize output array
    delta_smooth = np.zeros((len(bins), ngrid, ngrid, ngrid), dtype=np.float32)

    # Compute smoothed field for each R value in bins
    for i, R in enumerate(bins):
        # Defining a spherical shell window function in real space
        W = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
        coords = (np.arange(ngrid) - ngrid // 2) * grid_cell_size
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        r_grid = np.sqrt(X**2 + Y**2 + Z**2)

        W[(r_grid >= R - thickness / 2) & (r_grid <= R + thickness / 2)] = 1.0
        W /= np.sum(W)  # Normalize before FFT (avoids NaNs)
        W_shifted = np.fft.ifftshift(W)  # Center to corner for FFT alignment

        # Taking convolution of window function with overdensity field delta
        W_k = pyfftw.interfaces.numpy_fft.rfftn(W_shifted, threads=threads)
        delta_k_smooth = delta_k * W_k
        delta_smooth[i] = pyfftw.interfaces.numpy_fft.irfftn(delta_k_smooth, threads=threads) / np.sum(W)  
        # Normalized by number of points in the spherical shell, np.sum(W)
    
    # Interpolating the field at the tracer (galaxy) positions
    delta_interp = np.zeros((len(bins), len(pos)))  # Shape (number_of_bins, number_of_tracers)

    # Perform interpolation for each smoothed field
    for i, R in enumerate(bins):
        density_interpolated = np.zeros(pos.shape[0], dtype=np.float32)
        MASL.CIC_interp(delta_smooth[i], boxsize, pos.astype(np.float32), density_interpolated)
        delta_interp[i] = density_interpolated #delta_interp is a 10^5*11 array

    # Computing the 2-point Cross-Correlation Function by averaging over the interpolated field at the tracer positions
    xi = np.zeros_like(bins)
    for i, R in enumerate(bins):
        xi[i] = np.mean(delta_interp[i]) #averaging gives 10^5*11 to 11*1 array.
    
    return xi

#---------------------------------------------------------------------------------------------------