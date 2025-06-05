###############################################################################################################################

#-------------------------------------------  Importing the required libraries  -----------------------------------------------

import numpy as np
import pyfftw
from scipy import interpolate
import scipy.spatial
import time
import MAS_library as MASL
import smoothing_library as SL
from kNN_ASMR.HelperFunctions import create_smoothed_field_dict_3D

###############################################################################################################################

# Function that returns the the 2-point Cross-Correlation Function between a set of discrete tracers and a continuous field using the stacking method

def CrossCorr2pt(boxsize, bins, QueryPos, TracerPos, delta, thickness, BoxSize, R, kmin=None, kmax=None, Verbose=False):
    '''
    Calculates the Two-point Cross-correlation function between a set of tracers and a field. The interpolation can only be done using the 
    CIC-mass assignment scheme.

    Parameters
    ----------
    bins : float array of shape (m,)
        Set of m radial distances at which the 2PCF will be computed
        
    querypos : float array of shape (n_query,3) where n_query is the number of query points
            3D positions of the random query points inside the box, given in Cartesian coordinates (x, y, z) within the range         [0, boxsize]

    BoxSize : float
        The length (in Mpc/h) of the cubic box containing the tracers and the field

    pos : float array of shape (n_pos, 3) where n_pos is the number of discrete tracers
        3D positions of the n tracers (e.g., galaxies) inside the box, given in Cartesian coordinates (x, y, z) within the range         [0, boxsize]

    delta : float array of shape (ngrid, ngrid, ngrid)
        Smoothed overdensity field defined on a uniform grid with ngrid³ points

    thickness : float
        Thickness (in Mpc/h) of the spherical shell used for stacking to compute the 2PCF
    
    Returns
    -------
    xi : float array of shape (m,)
        The 2-point cross-correlation function (2PCF) between the tracer positions and the field, computed at each of the m radial bins.
    
    Raises
    ------
    ValueError
        if the given query points are not on a three-dimensional grid.
    ValueError
        if x,y, or z coordinate of any of the query points is not in ``(0, BoxSize)``.
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
    
    smoothed_delta_dict= create_smoothed_field_dict_3D(field=delta, Filter='Shell', grid=QueryPos, Boxsize=boxsize, bins=bins, kmin=kmin, kmax=kmax, thickness=thickness, Verbose=Verbose)
    
    delta_interp = np.zeros((len(bins), len(TracerPos)))  # Shape (number_of_bins, number_of_tracers)

    # Perform interpolation for each smoothed field
    for i, R in enumerate(bins):
        density_interpolated = np.zeros(TracerPos.shape[0], dtype=np.float32)
        MASL.CIC_interp(smoothed_delta_dict[str(R)][i], BoxSize, TracerPos, density_interpolated)
        delta_interp[i] = density_interpolated

    # Computing the 2-point Cross-Correlation Function by averaging over the interpolated field at the tracer positions
    xi = np.zeros_like(bins)
    for i, R in enumerate(bins):
        xi[i] = np.mean(delta_interp[i])
    
    return xi

#--------------------------------------------------------------------------------------------------------