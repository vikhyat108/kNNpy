####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import ICRS
import time
import copy
import sys
import os

#Necessary for relative imports (see https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im)
module_path = os.path.abspath(os.path.join(''))
'''
@private
'''

if module_path not in sys.path:
    sys.path.append(module_path)

#Importing the required helper function
from kNN_ASMR.HelperFunctions import create_smoothed_field_dict_2DA

####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

def CorrelationFunction(BinsRad, MaskedTracerPosRad, FieldSkymap, NR_ND=10, ReturnSmoothedDict=False, Verbose=False):
    
    r'''
    Computes the angular two-point cross-correlation function between the given set of discrete tracers (`MaskedTracerPosRad`) and the given continuous overdensity field (`FieldSkymap`) at the given angular distance scales (`BinsRad`).

    Parameters
    ----------
    BinsRad : list of numpy float array
        array of angular distances (in radians) to compute the cross-correlation function at
    MaskedTracerPosRad : numpy float array of shape ``(n_tracer, 2)``
        array of sky locations for the discrete tracers. For each data point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    FieldSkymap : numpy float array
        the healpy map of the continuous field. The values of the masked pixels, if any, should be set to `hp.UNSEEN`.
    NR_ND : int
        ratio of number of randoms to number of data points used in the 2PCF calculation to remove biases caused by the presence of an observational mask, by default ``10``. This is similar to the ratio of number of randoms to number data points used in the usual Landy-Szalay estimator of the 2PCF[^1]. See notes for a more detailed explanation of this parameter and how to set it appropriately.
    ReturnSmoothedDict : bool, optional
        if set to ``True``, the dictionary containing the continuous field masked within the observational footprint, and smoothed at the provided angular distance scales, will be returned along with the nearest-neighbour measurements, by default ``False``.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    w_theta: numpy float array of shape ``(len(Bins)-1, )``
        tracer-field two-point cross-correlation function evaluated at the desired distance bins. Note that the 2PCF can't be estimated at the last bin due to the nature of the algorithm (refer to notes for details).
    SmoothedFieldDict : dict
        dictionary containing the continuous field masked within the observational footprint and smoothed at the provided angular distance scales, returned only if `ReturnSmoothedDict` is ``True``. For example, ``SmoothedFieldDict['0.215']`` represents the continuous map smoothed at a scale of 0.215 radians.

    Raises
    ------
    ValueError
        if declination of any of the tracer points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the tracer points is not in ``[0, 2*np.pi]``.
    ValueError
        if the given tracer points are not on a two-dimensional grid.
    ValueError
        if the `NR_ND` is not an integer.

    See Also
    --------
    kNN_ASMR.Auxilliary.TPCF.TracerField2DA.CorrelationFunction_DataVector : computes a data vector of tracer-field two-point cross-correlations in 2D for multiple realisations of the tracer set.
    kNN_ASMR.Auxilliary.TPCF.3DTPCF_Tracer-Field.CrossCorr2pt : computes tracer-field two-point cross-correlations in 3D.
    kNN_ASMR.kNN_2D_Ang.TracerFieldCross2DA : computes 2D angular tracer-field cross-correlations using the $k$NN formalism.

    Notes
    -----
    Measures the angular two-point cross-correlation function (2pcf) between a set of discrete tracers and a continuous overdensity field using the spherical band-averaging method, as described in Gupta (2024)[^1]. This is a generalisation of the spherical shell-averaging method described in Banerjee & Abel (2023)[^2].
    
    Data with associated observational footprints are supported, in which case, only tracer positions within the footprint should be provided and the field should be masked appropriately. If the footprints of the tracer set and the field are different, a combined mask representing the intersection of the two footprints should be used. The field at the masked pixels is set to 0 (which is the mean value of an overdensity field) for the purposes of the 2PCF computation to prevent any biases due to the mask.

    <enter description for the NR_ND parameter>
    <enter description of algorithm and why the 2PCF can't be estimated at the last bin>

    References
    ----------
    [^1]: Kaustubh Rajesh Gupta, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, MS Thesis, Indian Institute of Science Education and Research Pune [Digital Repository](http://dr.iiserpune.ac.in:8080/xmlui/handle/123456789/8819)

    [^2]: Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour   distributions, [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stac3813), Volume 519, Issue 4, March 2023, Pages 4856–4868
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if np.any(MaskedTracerPosRad[:, 0]<-np.pi/2 or MaskedTracerPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid tracer point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedTracerPosRad[:, 1]<0 or MaskedTracerPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid tracer point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if MaskedTracerPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 2), where n_tracer is the number of tracers.')
    
    if not isinstance(NR_ND, int): 
        raise ValueError("Please input an integer value for 'NR_ND'.")

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #Getting the mask, sky coverage and HEALPix NSIDE

    mask = np.ones_like(FieldSkymap).astype(int)
    mask_ind = np.where(FieldSkymap==hp.UNSEEN)
    mask[mask_ind] = 0
    sky_frac = len(np.where(mask==1)[0])/len(mask)
    NSIDE = hp.get_nside(FieldSkymap)

    #-----------------------------------------------------------------------------------------------

    #Calculate the smoothed field dictionary

    if Verbose: 
        smooth_start_time = time.perf_counter()
        print('\nSmoothing the continuous field at the given angular distance scales...')

    #Note: QueryMask is set to 2 everywhere to avoid masking out any additional pixels. The pixels outside the footprint are automatically masked during the smoothing process (refer to the HelperFunctions module documentation for more details).
    SmoothedFieldDict = create_smoothed_field_dict_2DA(FieldSkymap, [BinsRad], QueryMask=2*np.ones_like(FieldSkymap).astype(int), Verbose=False)

    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-smooth_start_time))
    
    #-----------------------------------------------------------------------------------------------

    #Computing the 2pcf

    if Verbose: 
        tpcf_start_time = time.perf_counter()
        print('\nComputing the 2PCF...')

    w_theta = np.zeros(len(BinsRad)-1)
    
    for i, ss in enumerate(BinsRad[:-1]):

        sdgm_low = copy.deepcopy(SmoothedFieldDict[str(ss)])
        sdgm_low[sdgm_low==hp.UNSEEN] = 0
        sdgm_high = copy.deepcopy(SmoothedFieldDict[str(BinsRad[i+1])])
        sdgm_high[sdgm_high==hp.UNSEEN] = 0

        hp_object = HEALPix(nside=NSIDE, order='ring', frame=ICRS())

        #Discrete tracers
        ra = MaskedTracerPosRad[:, 1]*u.radian
        dec = MaskedTracerPosRad[:, 0]*u.radian
        coords = SkyCoord(ra, dec, frame='icrs')
        ds2 = hp_object.interpolate_bilinear_skycoord(coords, sdgm_high)
        ds1 = hp_object.interpolate_bilinear_skycoord(coords, sdgm_low)

        #Randoms
        n_rand = MaskedTracerPosRad.shape[0]*NR_ND
        ra_rand = np.random.uniform(low=0, high=2*np.pi, size=2*int(n_rand/sky_frac))
        dec_rand = np.arcsin(np.random.uniform(low=-1, high=1, size=2*int(n_rand/sky_frac)))
        ipix = hp.ang2pix(NSIDE, np.rad2deg(ra_rand), np.rad2deg(dec_rand), nest=False, lonlat=True)
        mask_val = mask[ipix]
        ind_masked = np.where(mask_val==1)[0]
        ra_rand_masked = ra_rand[ind_masked]
        dec_rand_masked = dec_rand[ind_masked]
        ind_ds = np.random.choice(len(ra_rand_masked), n_rand, replace=False)
        ra_rand_masked_ds = ra_rand_masked[ind_ds]*u.radian
        dec_rand_masked_ds = dec_rand_masked[ind_ds]*u.radian
        coords_rand = SkyCoord(ra_rand_masked_ds, dec_rand_masked_ds, frame='icrs')
        ds2_rand = hp_object.interpolate_bilinear_skycoord(coords_rand, sdgm_high)
        ds1_rand = hp_object.interpolate_bilinear_skycoord(coords_rand, sdgm_low)
        
        A2 = 2*np.pi*(1-np.cos(BinsRad[i+1]))
        A1 = 2*np.pi*(1-np.cos(BinsRad[i]))

        w_theta[i] = np.mean((A2*ds2-A1*ds1)/(A2-A1)) - np.mean((A2*ds2_rand-A1*ds1_rand)/(A2-A1))

    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-tpcf_start_time))

    #-----------------------------------------------------------------------------------------------

    if Verbose: print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))
    
    if ReturnSmoothedDict: 
        return w_theta, SmoothedFieldDict
    else: 
        return w_theta

####################################################################################################
    
def CorrelationFunction_DataVector(BinsRad, MaskedTracerPosVectorRad, FieldSkymap, NR_ND=10, ReturnSmoothedDict=False, Verbose=False):
    
    r'''
    Returns a 'data vector' of the angular two-point cross-correlation function between multiple realisations of the given set of discrete tracers (`MaskedTracerPosVectorRad`) and a single realisation of the given continuous overdensity field (`FieldSkymap`) at the given angular distance scales (`BinsRad`). Please refer to notes to understand why this might be useful.

    Parameters
    ----------
    BinsRad : list of numpy float array
        array of angular distances (in radians) to compute the cross-correlation function at
    MaskedTracerPosVectorRad : numpy float array of shape ``(n_realisations, n_tracer, 2)``
        array of sky locations for the first set of discrete tracers. For each data point in the array, the first (second) coordinate should be the declination (right ascension) in radians. Please ensure ``-np.pi/2 <= declination <= pi/2`` and ``0 <= right ascension <= 2*np.pi``.
    FieldSkymap : numpy float array
        the healpy map of the continuous field. The values of the masked pixels, if any, should be set to `hp.UNSEEN`.
    NR_ND : int
        ratio of number of randoms to number of data points used in the 2PCF calculation to remove biases caused by the presence of an observational mask, by default ``10``. This is similar to the ratio of number of randoms to number data points used in the usual Landy-Szalay estimator of the 2PCF[^1]. See notes for a more detailed explanation of this parameter and how to set it appropriately.
    ReturnSmoothedDict : bool, optional
        if set to ``True``, the dictionary containing the continuous field masked within the observational footprint, and smoothed at the provided angular distance scales, will be returned along with the nearest-neighbour measurements, by default ``False``.
    Verbose : bool, optional
        if set to ``True``, the time taken to complete each step of the calculation will be printed, by default ``False``.

    Returns
    -------
    w_theta_vector: numpy float array of shape ``(n_realisations, len(BinsRad)-1)``
        data vector containing the tracer-field two-point cross-correlation function for multiple realisations of the discrete tracer set evaluated at the desired distance bins. Note that the 2PCF can't be estimated at the last bin due to the nature of the algorithm (refer to notes for details).
    SmoothedFieldDict : dict
        dictionary containing the continuous field masked within the observational footprint and smoothed at the provided angular distance scales, returned only if `ReturnSmoothedDict` is ``True``. For example, ``SmoothedFieldDict['0.215']`` represents the continuous map smoothed at a scale of 0.215 radians.

    Raises
    ------
    ValueError
        if declination of any of the tracer points is not in ``[-np.pi/2, np.pi/2]``.
    ValueError
        if right ascension of any of the tracer points is not in ``[0, 2*np.pi]``.
    ValueError
        if the given tracer points are not on a two-dimensional grid.
    ValueError
        if the `NR_ND` is not an integer.

    See Also
    --------
    kNN_ASMR.Auxilliary.TPCF.TracerField2DA.CorrelationFunction : computes tracer-field two-point cross-correlations in 2D for a single realisation of the tracer set.
    kNN_ASMR.Auxilliary.TPCF.3DTPCF_Tracer-Field.CrossCorr2pt : computes tracer-field two-point cross-correlations in 3D.
    kNN_ASMR.kNN_2D_Ang.TracerFieldCross2DA_DataVector : computes a data vector of 2D angular tracer-field cross-correlations for multiple tracer realisations using the $k$NN formalism.

    Notes
    -----
    Please refer to the documentation of kNN_ASMR.Auxilliary.TPCF.TracerFieldCross2DA.CorrelationFunction for important usage notes that also apply to this function and references. <Explain why cross-correlating multiple realisations of tracer with single realisation of field might be useful>

    References
    ----------
    [^1]: Kaustubh Rajesh Gupta, Spatial clustering of gravitational wave sources with k-nearest neighbour distributions, MS Thesis, Indian Institute of Science Education and Research Pune [Digital Repository](http://dr.iiserpune.ac.in:8080/xmlui/handle/123456789/8819)
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if np.any(MaskedTracerPosVectorRad[:, :, 0]<-np.pi/2 or MaskedTracerPosVectorRad[:, :, 0]>np.pi/2):
        raise ValueError('Invalid tracer point position(s): please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedTracerPosVectorRad[:, :, 1]<0 or MaskedTracerPosVectorRad[:, :, 0]>2*np.pi):
        raise ValueError('Invalid tracer point position(s): please ensure 0 <= right ascension <= 2*pi.')

    if MaskedTracerPosVectorRad.shape[2]!=2: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer positions must be of shape (n_tracer, 2), where n_tracer is the number of tracers.')
    
    if not isinstance(NR_ND, int): 
        raise ValueError("Please input an integer value for 'NR_ND'.")

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------

    #Getting the mask, sky coverage and HEALPix NSIDE

    mask = np.ones_like(FieldSkymap).astype(int)
    mask_ind = np.where(FieldSkymap==hp.UNSEEN)
    mask[mask_ind] = 0
    sky_frac = len(np.where(mask==1)[0])/len(mask)
    NSIDE = hp.get_nside(FieldSkymap)

    #-----------------------------------------------------------------------------------------------

    #Calculate the smoothed field dictionary

    if Verbose: 
        smooth_start_time = time.perf_counter()
        print('\nSmoothing the continuous field at the given angular distance scales...')

    #Note: QueryMask is set to 2 everywhere to avoid masking out any additional pixels. The pixels outside the footprint are automatically masked during the smoothing process (refer to the HelperFunctions module documentation for more details).
    SmoothedFieldDict = create_smoothed_field_dict_2DA(FieldSkymap, [BinsRad], QueryMask=2*np.ones_like(FieldSkymap).astype(int), Verbose=False)

    if Verbose: print('\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-smooth_start_time))
    
    #-----------------------------------------------------------------------------------------------

    #Looping the 2PCF calculation over the multiple realisations

    n_reals = MaskedTracerPosVectorRad.shape[0]
    w_theta_vector = np.zeros((n_reals, len(BinsRad)-1))

    for realisation, MaskedTracerPosRad in enumerate(MaskedTracerPosVectorRad):

        if Verbose:
            start_time_real = time.perf_counter()
            print(f'\n\n--------------  Realisation {realisation+1}/{n_reals}  --------------\n')

        #-------------------------------------------------------------------------------------------

        #Computing the 2pcf

        if Verbose: 
            print('Computing the 2PCF...')

        w_theta_vector[realisation] = np.zeros(len(BinsRad)-1)
        
        for i, ss in enumerate(BinsRad[:-1]):

            sdgm_low = copy.deepcopy(SmoothedFieldDict[str(ss)])
            sdgm_low[sdgm_low==hp.UNSEEN] = 0
            sdgm_high = copy.deepcopy(SmoothedFieldDict[str(BinsRad[i+1])])
            sdgm_high[sdgm_high==hp.UNSEEN] = 0

            hp_object = HEALPix(nside=NSIDE, order='ring', frame=ICRS())

            #Discrete tracers
            ra = MaskedTracerPosRad[:, 1]*u.radian
            dec = MaskedTracerPosRad[:, 0]*u.radian
            coords = SkyCoord(ra, dec, frame='icrs')
            ds2 = hp_object.interpolate_bilinear_skycoord(coords, sdgm_high)
            ds1 = hp_object.interpolate_bilinear_skycoord(coords, sdgm_low)

            #Randoms
            n_rand = MaskedTracerPosRad.shape[0]*NR_ND
            ra_rand = np.random.uniform(low=0, high=2*np.pi, size=2*int(n_rand/sky_frac))
            dec_rand = np.arcsin(np.random.uniform(low=-1, high=1, size=2*int(n_rand/sky_frac)))
            ipix = hp.ang2pix(NSIDE, np.rad2deg(ra_rand), np.rad2deg(dec_rand), nest=False, lonlat=True)
            mask_val = mask[ipix]
            ind_masked = np.where(mask_val==1)[0]
            ra_rand_masked = ra_rand[ind_masked]
            dec_rand_masked = dec_rand[ind_masked]
            ind_ds = np.random.choice(len(ra_rand_masked), n_rand, replace=False)
            ra_rand_masked_ds = ra_rand_masked[ind_ds]*u.radian
            dec_rand_masked_ds = dec_rand_masked[ind_ds]*u.radian
            coords_rand = SkyCoord(ra_rand_masked_ds, dec_rand_masked_ds, frame='icrs')
            ds2_rand = hp_object.interpolate_bilinear_skycoord(coords_rand, sdgm_high)
            ds1_rand = hp_object.interpolate_bilinear_skycoord(coords_rand, sdgm_low)
            
            A2 = 2*np.pi*(1-np.cos(BinsRad[i+1]))
            A1 = 2*np.pi*(1-np.cos(BinsRad[i]))

            w_theta_vector[realisation, i] = np.mean((A2*ds2-A1*ds1)/(A2-A1)) - np.mean((A2*ds2_rand-A1*ds1_rand)/(A2-A1))

        if Verbose: print('\tdone. time taken for realisation {}: {:.2e} s.'.format(realisation+1, time.perf_counter()-start_time_real))

    #-----------------------------------------------------------------------------------------------

    if Verbose: print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))
    
    if ReturnSmoothedDict: 
        return w_theta_vector, SmoothedFieldDict
    else: 
        return w_theta_vector

####################################################################################################

#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################