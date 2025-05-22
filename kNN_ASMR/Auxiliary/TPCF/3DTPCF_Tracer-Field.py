###############################################################################################################################

#-------------------------------------------  Importing the required libraries  -----------------------------------------------

import numpy as np
import pyfftw
from scipy import interpolate
import scipy.spatial
import time
import MAS_library as MASL
import smoothing_library as SL
from HelperFunctions.py import smoothing_3D, create_query_3D

###############################################################################################################################

# Function that returns the the 2-point Cross-Correlation Function between a set of discrete tracers and a continuous field using the stacking method

def CrossCorr2pt(bins, pos, delta, thickness, Filter, BoxSize, R=None, kmin=None, kmax=None, Verbose=False):
    '''
    Calculates the Two-point Cross-correlation function between a set of tracers and a field. The interpolation can only be done using the 
    CIC-mass assignment scheme.

    Parameters
    ----------
    bins : float array of shape (m,)
        Set of m radial distances at which the 2PCF will be computed
        
    BoxSize : float
        The length (in Mpc/h) of the cubic box containing the tracers and the field

    pos : float array of shape (n, 3)
        3D positions of the n tracers (e.g., galaxies) inside the box, given in Cartesian coordinates (x, y, z) within the range         [0, boxsize]

    delta : float array of shape (ngrid, ngrid, ngrid)
        Smoothed overdensity field defined on a uniform grid with ngrid³ points

    thickness : float
        Thickness (in Mpc/h) of the spherical shell used for stacking to compute the 2PCF
    
    Returns
    -------
    xi : float array of shape (m,)
        The 2-point cross-correlation function (2PCF) between the tracer positions and the field, computed at each of the m radial bins.
    '''

    # Calculating the number of grid points along each axis of the field delta
    shape = np.shape(delta)
    if len(shape) != 3 or shape[0] != shape[1] or shape[1] != shape[2]:
        raise ValueError("Error: Input array is not cubical (n, n, n).")
    ngrid = shape[0]

    grid = np.linspace(0, BoxSize, ngrid, endpoint=False)  # Grid points in each dimension
    smoothed_delta= smoothing_3D(delta, Filter, grid, BoxSize, R, kmin, kmax, thickness, Verbose)
    
    delta_interp = np.zeros((len(bins), len(pos)))  # Shape (number_of_bins, number_of_tracers)

    # Perform interpolation for each smoothed field
    for i, R in enumerate(bins):
        density_interpolated = np.zeros(pos.shape[0], dtype=np.float32)
        MASL.CIC_interp(smoothed_delta[i], BoxSize, pos, density_interpolated)
        delta_interp[i] = density_interpolated

    # Computing the 2-point Cross-Correlation Function by averaging over the interpolated field at the tracer positions
    xi = np.zeros_like(bins)
    for i, R in enumerate(bins):
        xi[i] = np.mean(delta_interp[i])
    
    return xi

#--------------------------------------------------------------------------------------------------------