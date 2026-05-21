####################################################################################################

#-------------------------------  Importing the required libraries  --------------------------------

from venv import logger

import numpy as np
import time
import pyfftw
import MAS_library as MASL
import sys
import os
from kNNpy.HelperFunctions import compute_mean_parallel
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

def CrossCorr2pt(bins, pos, delta, boxsize, threads, W_k_list):
    """
    Compute 2-point cross-correlation function between tracers and density field.
    Optimized: avoid memory copies, use in-place operations, reuse arrays.
    
    Parameters:
    -----------
    bins : array
        Radial bins for correlation function
    boxsize : float
        Size of simulation box
    pos : array
        Tracer positions (N, 3), must be float32
    delta : array
        Density field (ngrid, ngrid, ngrid)
    threads : int
        Number of threads for FFT
        
    Returns:
    --------
    xi : array
        Cross-correlation function values
    """
    # Calculating the number of grid points along each axis of the field delta
    shape = np.shape(delta)
    if len(shape) != 3 or shape[0] != shape[1] or shape[1] != shape[2]:
        raise ValueError("Error: Input array is not cubical (n, n, n).")

    # Fourier Transforming the delta overdensity field once
    delta_k = pyfftw.interfaces.numpy_fft.rfftn(delta, threads=threads)
    
    # Ensure positions are float32 to avoid repeated casting
    if pos.dtype != np.float32:
        pos = pos.astype(np.float32)

    # Compute 2PCF by processing each smoothing scale
    xi = np.zeros(len(bins), dtype=np.float32)
    
    for i, W_k in enumerate(W_k_list):
        # Apply filter in Fourier space
        delta_k_smooth = delta_k * W_k
        # Transform back to real space
        delta_smooth = pyfftw.interfaces.numpy_fft.irfftn(delta_k_smooth, threads=threads)
        
        # Interpolate to particle positions
        density_interpolated = np.zeros(pos.shape[0], dtype=np.float32)
        MASL.CIC_interp(delta_smooth, boxsize, pos, density_interpolated)
        
        # Average to get correlation function value (using Numba-optimized mean)
        xi[i] = compute_mean_parallel(density_interpolated)
    
    return xi
#---------------------------------------------------------------------------------------------------