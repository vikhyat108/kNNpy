####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

from tabnanny import verbose

import numpy as np

import healpy as hp

import pandas as pd
import readgadget
from astrotools import healpytools as hpt
import MAS_library as MASL


import os
import sys

#Necessary for relative imports (see https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im)
module_path = os.path.abspath(os.path.join('../'))
'''
@private
'''
if module_path not in sys.path:
    sys.path.append(module_path)

from kNNpy.HelperFunctions_2DA import cartesian_corner_to_angles_centre, cat2hpx

####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

####################################################################################################

def Load_forecast_LSSTy1_galaxy_field(masked=False, NSIDE=512, DataPath='../kNNpy/Data'):
    '''
    Loads a forecast galaxy overdensity field for the Rubin Telescope's LSST Y1[^1] data release, created using the Agora simulations[^2]. Please refer to Gupta (2024)[^3] for details on the LSST forecast.

    Parameters
    ----------
    masked : bool, optional
        whether to load the masked overdensity field, by default, loads the unmasked version. Set to ``True`` to load the field masked using a realistic forecast for the LSST y1 observational footprint.

    NSIDE : int, optional
        HEALPix nside of the target map, by default 512.

    DataPath : str, optional
        path to the kNNpy Data directory, by default '../kNNpy/Data/'.

    Returns
    -------
    delta_gal_masked : float ndarray
        the HEALPix array containg the forecast LSST y1 galaxy overdensity field.

    mask : int ndarray
        only returned if ``masked=True``; the HEALPix mask defining the LSST y1 survey footprint. Pixels within the footprint have value 1, while pixels outside the footprint have value ``hp.UNSEEN``.

    Notes
    -----
    Currently, only an ``NSIDE=512`` field is supported.

    References
    ----------
    [^1]: <add reference here>
    [^2]: <add reference here>
    [^3]: <add reference here>
    '''

    #-----------------------------------------------------------------------------------------------

    fol_dataset = f'{DataPath}/Agora_simulations/LSST_forecast_lightcone'

    if masked:
        delta_gal_masked = np.load(f'{fol_dataset}/LSST_y1_forecast_Agora_lightcone_galaxy_overdensity_field_NSIDE_{NSIDE}_LSST_forecast_mask.npy')
        mask = np.load(f'{fol_dataset}/LSST_mask_NSIDE_{NSIDE}_abs_glat_max_15_dec_-70_to_12.5.npy')
        return delta_gal_masked, mask
    else:
        delta_gal_masked = np.load(f'{fol_dataset}/LSST_y1_forecast_Agora_lightcone_galaxy_overdensity_field_NSIDE_{NSIDE}_unmasked.npy')
        return delta_gal_masked

####################################################################################################

def Load_WSC_mask(NSIDE=256, DataPath='../kNNpy/Data'):
    '''
    Loads WISExSCOS[^1] survey footprint mask.

    Parameters
    ----------
    NSIDE : int, optional
        HEALPix nside of the target map, by default 256.

    DataPath : str, optional
        path to the kNNpy Data directory, by default '../kNNpy/Data/'.

    Returns
    -------
    mask : int ndarray
        the HEALPix mask defining the WISExSCOS survey footprint. Pixels within the footprint have value 1, while pixels outside the footprint have value ``hp.UNSEEN``.

    Notes
    -----
    Currently, only an ``NSIDE=256`` mask is supported.

    References
    ----------
    [^1]: <add reference here>
    '''

    #-----------------------------------------------------------------------------------------------

    fol_dataset = f'{DataPath}/WISExSCOS'
    mask = np.load(f'{fol_dataset}/WISExSCOSmask_NSIDE_{NSIDE}_equatorial_coords.npy')
    
    return mask

####################################################################################################

def Sample2DTracersFromField(delta_sampling, mask, N_realisations, n_tracers, seed=None, map_NSIDE=64):
    '''
    Creates a number of realisations of discrete tracers sampled from a given 2D angular overdensity field using Poisson sampling.

    Parameters
    ----------
    delta_sampling : float ndarray
        HEALPix overdensity field from which tracers are to be sampled.

    mask : int ndarray
        HEALPix mask defining the survey footprint. Pixels within the footprint should have value 1, while pixels outside the footprint should have value ``hp.UNSEEN``.

    N_realisations : int
        total number of realisations to be sampled.

    n_tracers : int
        total number of tracers to be sampled.

    seed : int, optional
        random seed for reproducibility, by default None.

    map_NSIDE : int, optional
        HEALPix nside of the output number counts map, by default 64.

    Returns
    -------
    
    tracer_pos_masked_ds_arr : float ndarray
        array of shape (``N_realisations``, ``n_tracers``, 2) containing the sampled tracer positions in radians. The last dimension contains (Dec, RA) pairs.

    map : float ndarray
        HEALPix map of the number counts of the sampled tracers, summed over all realisations.
    '''

    #-----------------------------------------------------------------------------------------------

    NSIDE = hp.npix2nside(len(delta_sampling))
    Nobj = int(1.5*n_tracers*12*NSIDE**2/len(np.where(mask==1)[0]))

    tracer_pos_arr = np.zeros((N_realisations, Nobj, 2))
    tracer_pos_masked_ds_arr = np.zeros((N_realisations, n_tracers, 2))

    np.random.seed(seed)

    for i, tracer_pos in enumerate(tracer_pos_arr):

        sampled_pix = hpt.rand_pix_from_map(1+delta_sampling, n=Nobj)
        sampled_RAs, sampled_Decs = hp.pix2ang(NSIDE, sampled_pix, lonlat=True)
        tracer_pos[:, 0] = np.deg2rad(sampled_Decs)
        tracer_pos[:, 1] = np.deg2rad(sampled_RAs)
        ipix = hp.ang2pix(NSIDE, np.rad2deg(tracer_pos[:, 1]), np.rad2deg(tracer_pos[:, 0]), nest=False, lonlat=True)
        mask_val = mask[ipix]
        ind_masked = np.where(mask_val!=hp.UNSEEN)[0]
        tracer_pos_masked = tracer_pos[ind_masked]
        # print(len(tracer_pos_masked), n_tracers)
        tracer_pos_masked_ds_arr[i] = tracer_pos_masked[np.random.choice(len(tracer_pos_masked), n_tracers, replace=False)]

    map = cat2hpx(np.rad2deg(tracer_pos_masked_ds_arr)[:, :, 1], np.rad2deg(tracer_pos_masked_ds_arr)[:, :, 0], nside=map_NSIDE, radec=False).astype(float)

    return tracer_pos_masked_ds_arr, map

####################################################################################################

def Sample2DPoissonTracers(mask, N_realisations, n_tracers, seed=None, map_NSIDE=64):
    '''
    Creates a number of realisations of discrete tracers sampled from a uniform Poisson distribution over the sky.

    Parameters
    ----------
    mask : int ndarray
        HEALPix mask defining the survey footprint. Pixels within the footprint should have value 1, while pixels outside the footprint should have value ``hp.UNSEEN``.

    N_realisations : int
        total number of realisations to be sampled.

    n_tracers : int
        total number of tracers to be sampled.

    seed : int, optional
        random seed for reproducibility, by default None.

    map_NSIDE : int, optional
        HEALPix nside of the output number counts map, by default 64.

    Returns
    -------
    
    randoms_pos_masked_ds_arr : float ndarray
        array of shape (``N_realisations``, ``n_tracers``, 2) containing the sampled tracer positions in radians. The last dimension contains (Dec, RA) pairs.

    map : float ndarray
        HEALPix map of the number counts of the sampled tracers, summed over all realisations.
    '''

    #-----------------------------------------------------------------------------------------------

    NSIDE = hp.npix2nside(len(mask))
    Nobj = int(1.5*n_tracers*12*NSIDE**2/len(np.where(mask==1)[0]))

    randoms_pos_arr = np.zeros((N_realisations, Nobj, 2))
    randoms_pos_masked_ds_arr = np.zeros((N_realisations, n_tracers, 2))

    np.random.seed(seed)

    for i, randoms_pos in enumerate(randoms_pos_arr):

        randoms_pos[:, 0] = np.arcsin(np.random.uniform(low=-1, high=1, size=randoms_pos.shape[0]))
        randoms_pos[:, 1] = np.random.uniform(low=0, high=2*np.pi, size=randoms_pos.shape[0])
        ipix = hp.ang2pix(NSIDE, np.rad2deg(randoms_pos[:, 1]), np.rad2deg(randoms_pos[:, 0]), nest=False, lonlat=True)
        mask_val = mask[ipix]
        ind_masked = np.where(mask_val!=hp.UNSEEN)[0]
        randoms_pos_masked = randoms_pos[ind_masked]
        randoms_pos_masked_ds_arr[i] = randoms_pos_masked[np.random.choice(len(randoms_pos_masked), n_tracers, replace=False)]

    map = cat2hpx(np.rad2deg(randoms_pos_masked_ds_arr)[:, :, 1], np.rad2deg(randoms_pos_masked_ds_arr)[:, :, 0], nside=map_NSIDE, radec=False).astype(float)

    return randoms_pos_masked_ds_arr, map

####################################################################################################

def Sample3DPoissonTracers(N_realisations, n_tracers, boxsize=1000, starting_seed=42):
    '''
    Creates a number of realisations of discrete tracers sampled from a uniform Poisson distribution in 3D space.

    Parameters
    ----------
    N_realisations : int
        total number of realisations to be sampled.

    n_tracers : int
        total number of tracers to be sampled.

    boxsize : float, optional
        size of the cubic box in which the tracers are to be sampled, by default 1000.0 Mpc/h.

    starting_seed : int, optional
        random seed for reproducibility, by default 42. This is the seed set for the first realisation,
        Henceforth, the next ith realisation has seed = starting_seed + i.

    Returns
    -------
    
    randoms_pos_ds_arr : float ndarray
        array of shape (``N_realisations``, ``n_tracers``, 3) containing the sampled tracer positions in Mpc/h.
    '''

    #-----------------------------------------------------------------------------------------------

    randoms_pos_ds_arr = []

    np.random.seed(starting_seed)

    for i in range(N_realisations):

        randoms_pos_ds_arr.append(np.random.uniform(low=0, high=boxsize, size=(n_tracers, 3)))
    
    randoms_pos_ds_arr = np.array(randoms_pos_ds_arr)

    return randoms_pos_ds_arr

################################################################################################

def Sample2DTracersFromQuijoteBox(sim_num, tracer_type, mask, N_realisations, n_tracers, seed=None, map_NSIDE=64, DataPath='../kNNpy/Data'):
    '''
    Creates a number of realisations of discrete tracers sampled from dark-matter halo catalogues created using the Quijote simulation hight-resolution boxes[^1]. The 3D tracer positions are converted to angular 2D coordinates by projecting them onto the sky of a hypothetical observer at the centre of the simulation box. Currently supports two types of tracers, namely 'Galaxies' and 'Clusters' delineated based on a simple mass cut; galaxies are halos with mass ``M_halo >= 3e13 Msun/h and M_halo <= 1e14 Msun/h``, while clusters are halos with mass ``M_halo >= 1e14 Msun/h and M_halo <= 2.5e14 Msun/h``.

    Parameters
    ----------
    sim_num : int
        Quijote simulation realisation number to be used for sampling the tracers.

    tracer_type : str
        type of tracers to be sampled. Currently supports 'Galaxies' and 'Clusters'.

    mask : int ndarray
        HEALPix mask defining the survey footprint. Pixels within the footprint should have value 1, while pixels outside the footprint should have value ``hp.UNSEEN``.

    N_realisations : int
        total number of realisations to be sampled.

    n_tracers : int
        total number of tracers to be sampled. Must be less than the total number of tracers available in the given simulation box after projecting onto the virtual sky and applying the survey mask. If the observational footprint covers the entire sky, the maximum number of 'Clusters' ('Galaxies') that can be sampled is <enter limit here> (<enter limit here>). For partial sky coverage, this number might be significantly lower.

    seed : int, optional
        random seed for reproducibility, by default None.

    map_NSIDE : int, optional
        HEALPix nside of the output number counts map, by default 64.

    DataPath : str, optional
        path to the kNNpy Data directory, by default '../kNNpy/Data/'.

    Returns
    -------
    
    tracer_pos_masked_ds_arr : float ndarray
        array of shape (``N_realisations``, ``n_tracers``, 2) containing the sampled tracer positions in radians. The last dimension contains (Dec, RA) pairs.

    map : float ndarray
        HEALPix map of the number counts of the sampled tracers, summed over all realisations.

    References
    ----------
    [^1]: <add reference here>
    '''

    #-----------------------------------------------------------------------------------------------

    #Defining the quijote box size
    boxsize=1000.0  #Mpc/h

    #Converting the simulation number into a simulation name string. For example, when sim_num is 5, name will be 'sim_005' and when sim_num is 50, name will be 'sim_050'
    num = str(sim_num).zfill(3)

    #Getting the NSIDE from the mask
    NSIDE = hp.npix2nside(len(mask))
    Nobj = int(1.5*n_tracers*12*NSIDE**2/len(np.where(mask==1)[0]))

    if tracer_type=='Galaxies': 
        
        #Reading in the halo catalogs corresponding to the given simulation number
        folder_ga = f'{DataPath}/Quijote_simulations/Fiducial_HR_rockstar/Galaxy_cat'
        filepath_ga = '{a}/sim_{b}'.format(a = folder_ga, b = num)
        ga_data_selected = pd.read_csv(filepath_ga)

        #Positions of halos
        all_ga_pos = np.zeros((len(ga_data_selected), 3), dtype = np.float32)
        all_ga_pos[:, 0] = ga_data_selected.loc[:, '8'].values
        all_ga_pos[:, 1] = ga_data_selected.loc[:, '9'].values
        all_ga_pos[:, 2] = ga_data_selected.loc[:, '10'].values

        #Converting 3D positions to sky angles, observer at centre of box
        all_angpos = cartesian_corner_to_angles_centre(all_ga_pos[:, 0], all_ga_pos[:, 1], all_ga_pos[:, 2], boxsize, boxsize/2)


    elif tracer_type=='Clusters':
        
        #Reading in the halo catalogs corresponding to the given simulation number
        folder_cl = f'{DataPath}/Quijote_simulations/Fiducial_HR_rockstar/Cluster_cat'
        filepath_cl = '{a}/sim_{b}'.format(a = folder_cl, b = num)
        cl_data_selected = pd.read_csv(filepath_cl)

        #Positions of halos
        all_cl_pos = np.zeros((len(cl_data_selected), 3), dtype = np.float32)
        all_cl_pos[:, 0] = cl_data_selected.loc[:, '8'].values
        all_cl_pos[:, 1] = cl_data_selected.loc[:, '9'].values
        all_cl_pos[:, 2] = cl_data_selected.loc[:, '10'].values

        #Converting 3D positions to sky angles, observer at centre of box
        all_angpos = cartesian_corner_to_angles_centre(all_cl_pos[:, 0], all_cl_pos[:, 1], all_cl_pos[:, 2], boxsize, boxsize/2)

    else:
        raise ValueError("Invalid tracer type specified. Supported types are 'Galaxies' and 'Clusters'.")
    
    tracer_pos_arr = np.zeros((N_realisations, Nobj, 2))
    tracer_pos_masked_ds_arr = np.zeros((N_realisations, n_tracers, 2))

    np.random.seed(seed)

    for i, tracer_pos in enumerate(tracer_pos_arr):

        sel_ind = np.random.choice(len(all_angpos), Nobj, replace=False)
        tracer_pos = all_angpos[sel_ind]
        ipix = hp.ang2pix(NSIDE, np.rad2deg(tracer_pos[:, 1]), np.rad2deg(tracer_pos[:, 0]), nest=False, lonlat=True)
        mask_val = mask[ipix]
        ind_masked = np.where(mask_val!=hp.UNSEEN)[0]
        tracer_pos_masked = tracer_pos[ind_masked]
        tracer_pos_masked_ds_arr[i] = tracer_pos_masked[np.random.choice(len(tracer_pos_masked), n_tracers, replace=False)]

    map = cat2hpx(np.rad2deg(tracer_pos_masked_ds_arr)[:, :, 1], np.rad2deg(tracer_pos_masked_ds_arr)[:, :, 0], nside=map_NSIDE, radec=False).astype(float)

    return tracer_pos_masked_ds_arr, map

####################################################################################################

def Sample3DTracersFromQuijoteBox(tracer_type, N_realisations, n_tracers, ptype=[1],DataPath='../kNNpy/Data', starting_seed=42):
    '''
    For now it only supports one tracer_type: 'particles'. Other types like 'halos', can be included if and when necessry.
    ptype is an optional parameter only required for the 'particles' tracer type.
    '''
    np.random.seed(starting_seed)
    pos_array=[]
    if tracer_type=='particles':
        for i in range(N_realisations):
            snapshot=f'{DataPath}/Quijote_simulations/fiducial_LR/{i}/snapdir_004/snap_004'
            pos = readgadget.read_block(snapshot, 'POS ', ptype)/1e3 # Mpc/h
            pos = pos[np.random.choice(len(pos), n_tracers, replace=False)]
            pos_array.append(pos)
        pos_array=np.array(pos_array)
    else: 
        raise ValueError("Invalid tracer type specified. Supported types are 'Halos' and 'Particles'.")
    return pos_array

#########################################################################################################

def make_overdensity_3D(N_realisations, grid, ptype, do_RSD=False, MAS='CIC', axis=0, verbose=False, DataPath='../kNNpy/Data'):
    overdensity_list=[]
    for i in range(N_realisations):
        snapshot = f'{DataPath}/Quijote_simulations/fiducial_LR/{i}/snapdir_004/snap_004'
        delta = MASL.density_field_gadget(snapshot, ptype, grid, MAS, do_RSD, axis, verbose)
        delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
        overdensity_list.append(delta)
    overdensity_list=np.array(overdensity_list)
    return overdensity_list
#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################
