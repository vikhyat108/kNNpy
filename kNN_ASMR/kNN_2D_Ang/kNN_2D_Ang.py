####################################################################################################

#-------------------  These libraries are required for evaluating the functions  -------------------

import numpy as np
from sklearn.neighbors import BallTree
import time

from kNN_ASMR.HelperFunctions import calc_kNN_CDF

####################################################################################################

#--------------------------------------  Function Definitions  -------------------------------------

def TracerFieldCross2DA(kMax, BinsRad, MaskedQueryPosRad, MaskedTracerPosRad, SmoothedFieldDict,
                        FieldConstPercThreshold, Verbose=False):

    '''
    Measures the angular cross-correlation between a set of discrete tracers and a continuous 
    overdensity field using the k-nearest neighbour (kNN) formalism as defined in Banerjee & 
    Abel (2023) and Gupta & Banerjee (2024). 

    The field must already be smoothed at the desired angular distance scales using a top-hat 
    filter (see the 'HelperFunctions' module for help with smoothing). The smoothed fields need 
    to be provided as a dictionary ('SmoothedFieldDict'), see below for further details.
    
    Currently, the algorithm requires a constant percentile overdensity threshold for the 
    continuous field and query points to be defined on a HEALPix grid. Extentions to a constant 
    mass threshold and poisson-sampled query points on the sky may be added in the future. 
    
    Data with associated observational footprints are supported, in which case, only tracer 
    positions within the footprint should be provided and the field should be masked 
    appropriately. Importantly, in this case, query points need to be within the footprint and 
    appropriately padded from the edges of the footprint (see Gupta & Banerjee (2024) for a 
    detailed discussion). If the footprints of the tracer set and the field are different, a 
    combined mask representing the intersection of the two footprints should be used (see the 
    'HelperFunctions' module for help with masking and creating the modified query positions).

    Returns the probabilities P_{>k}, P_{>dt} and P_{>k,>dt} for 1 <= k <= kMax, that quantify the 
    extent of the spatial cross-correlation between the given discrete tracer positions 
    ('MaskedTracerPosRad') and the given continuous overdensity field. 
    	
        1. P_{>k}(theta): 
    		the kNN-CDF of the discrete tracers, evaluated at angular distance scale 'theta'
    		
    	2. P_{>dt}(theta): 
    		the probability of the overdensity field smoothed with a top-hat filter of angular 
            size 'theta' exceeding the given constant percentile density threshold
    		
    	3. P_{>k, dt}(theta):
    		the joint probability of finding at least 'k' tracers within a spherical cap of radius
            'theta' AND the overdensity field smoothed at angular scale 'theta' exceeding the given 
            density threshold
    		
    The excess cross-correlation (Banerjee & Abel 2023) can be computed trivially from these 
    quatities (see the 'HelperFunctions' module for a method to do this):
    	
    	Psi_{k, dt} = P_{>k, dt}/(P_{>k}*P_{>dt})

    #-----------------------------------------------------------------------------------------------

    References:

        1. Banerjee & Abel (2023):
        Arka Banerjee, Tom Abel, Tracer-field cross-correlations with k-nearest neighbour 
        distributions, Monthly Notices of the Royal Astronomical Society, Volume 519, Issue 4, 
        March 2023, Pages 4856–4868, https://doi.org/10.1093/mnras/stac3813
        
        2. Gupta & Banerjee (2024):
        Kaustubh Rajesh Gupta, Arka Banerjee, Spatial clustering of gravitational wave sources 
        with k-nearest neighbour distributions, Monthly Notices of the Royal Astronomical Society, 
        Volume 531, Issue 4, July 2024, Pages 4619–4639, https://doi.org/10.1093/mnras/stae1424

    #-----------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    
    kMax: int
        the number of nearest neighbours to calculate the distances to. For example, if kMax = 3, 
        the first 3 nearest-neighbour distributions will be computed.

    BinsRad: list of float arrays
        list of angular distances (in radians) for each nearest neighbour. The i^th element of the 
        list should contain a numpy array of the desired distances for the i^th nearest neighbour.

    MaskedQueryPosRad: float array of shape (n_query, 2)
        array of sky locations for the query points. The sky locations must be on a grid. For each
        query point in the array, the first (second) coordinate should be the declination (right
        ascension) in radians. 
        Please ensure -pi/2 <= declination <= pi/2 and 0 <= right ascension <= 2*pi.

    MaskedTracerPosRad: float array of shape (n_tracer, 2)
        array of sky locations for the discrete tracers. For each query point in the array, the 
        first (second) coordinate should be the declination (right ascension) in radians.
        Please ensure -pi/2 <= declination <= pi/2 and 0 <= right ascension <= 2*pi.

    SmoothedFieldDict: dictionary
        dictionary containing the continuous field masked within the observational footprint and 
        smoothed at various angular distance scales. For example, SmoothedFieldDict['0.215'] 
        represents the continuous map smoothed at a scale of 0.215 radians.

    FieldConstPercThreshold: float
        the percentile value for the constant percentile threshold to be used for the continuous 
        field. For example, FieldConstPercThreshold = 75.0 represents a 75th percentile threshold

    Verbose: Binary
        if set to True, the time taken to complete each step of the calculation will be printed.
        Defaults to 'False'

    #-----------------------------------------------------------------------------------------------

    Returns
    -------

    p_gtr_k_list: list of float arrays, each of shape len(BinsRad[k-1]) for 1 <= k <= kMax
        auto kNN-CDFs of the discrete tracers evaluated at the desired distance bins.
        
    p_gtr_dt_list: list of float arrays, each of shape len(BinsRad[k-1]) for 1 <= k <= kMax
        continuum version of auto kNN-CDFs for the continuous field evaluated at the desired 
        distance bins.
    
    p_gtr_k_dt_list: list of float arrays, each of shape len(BinsRad[k-1]) for 1 <= k <= kMax
        joint tracer-field nearest neighbour distributions evaluated at the desired distance bins
    '''
    
    #-----------------------------------------------------------------------------------------------

    if Verbose: total_start_time = time.perf_counter()

    #-----------------------------------------------------------------------------------------------
        
    #Step 0: Check all inputs are consistent with the function requirement

    if Verbose: print('Checking inputs ...')

    if MaskedQueryPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for query points: array containing the 
        query point positions must be of shape (n_query, 2), where n_query is the number of 
        query points.')

    if np.any(MaskedQueryPosRad[:, 0]<-np.pi/2 or MaskedQueryPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid query point position(s): 
        please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedQueryPosRad[:, 1]<0 or MaskedQueryPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid query point position(s): 
        please ensure 0 <= right ascension <= 2*pi.')

    if np.any(MaskedTracerPosRad[:, 0]<-np.pi/2 or MaskedTracerPosRad[:, 0]>np.pi/2):
        raise ValueError('Invalid tracer point position(s): 
        please ensure -pi/2 <= declination <= pi/2.')

    if np.any(MaskedTracerPosRad[:, 1]<0 or MaskedTracerPosRad[:, 0]>2*np.pi):
        raise ValueError('Invalid tracer point position(s): 
        please ensure 0 <= right ascension <= 2*pi.')

    if MaskedTracerPosRad.shape[1]!=2: 
        raise ValueError('Incorrect spatial dimension for tracers: array containing the tracer 
        positions must be of shape (n_tracer, 2), where n_tracer is the number of tracers.')

    if Verbose: print('\tdone.')

    #-----------------------------------------------------------------------------------------------
        
    #Step 1: Calculate nearest neighbour distances of query points, and the kNN-CDFs for the 
    #discrete tracers

    if Verbose: 
        step_1_start_time = time.perf_counter()
        print('\ninitiating step 1 ...')

    #-----------------------------------------------------------------------------------------------
    
    #Building the tree
    if Verbose: 
        start_time = time.perf_counter()
        print('\n\tbuilding the tree ...')
    xtree = BallTree(MaskedTracerPosRad, metric='haversine')
    if Verbose: print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the NN distances
    if Verbose: 
        start_time = time.perf_counter()
        print('\n\tcomputing the tracer NN distances ...')
    vol, _ = xtree.query(MaskedQueryPosRad, k=kMax)
    if Verbose: print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))

    #-----------------------------------------------------------------------------------------------
    
    #Calculating the kNN-CDFs
    if Verbose: 
        start_time = time.perf_counter()
        print('\n\tcomputing the tracer auto-CDFs P_{>k} ...')
    kList = range(1, kNN+1)
    p_gtr_k_list = calc_kNN_CDF(vol, kList, BinsRad)

    #-----------------------------------------------------------------------------------------------
    
    if Verbose: 
        print('\t\tdone; time taken: {:.2e} s.'.format(time.perf_counter()-start_time))
        print('\ntime taken for step 1: {:.2e} s.'.format(time.perf_counter()-step_1_start_time))
    
    #-----------------------------------------------------------------------------------------------
        
    #Steps 2: calculate the fraction of query points with nearest neighbour distance less than the 
    #angular distance and smoothened field greater than the overdensity threshold

    if Verbose: 
        step_2_start_time = time.perf_counter()
        print('\ninitiating step 2 ...')

    #-----------------------------------------------------------------------------------------------
        
    p_gtr_k_dt_list = []
    p_gtr_dt_list = []
    
    for k in range(kMax):

        if Verbose: 
            start_time = time.perf_counter()
            print('\n\tComputing P_{>dt} and P_{>k, dt} for k = {} ...'.format(k))

        p_gtr_k_dt = np.zeros(len(BinsRad[k]))
        p_gtr_dt = np.zeros(len(BinsRad[k]))

        for i, ss in enumerate(BinsRad[k]):

            #---------------------------------------------------------------------------------------

            #Load the smoothed field
            SmoothedField = SmoothedFieldDict[str(ss)]

            #Check if the masked smoothed field is consistent with the number of query points
            if SmoothedField.shape[0]!=MaskedQueryPosRad.shape[0]:
                raise ValueError('Shape of field smoothed at scale {:.3e} rad does not match shape 
                of query point array.'.format(ss))

            #---------------------------------------------------------------------------------------
            
            #Compute the overdensity threshold            
            delta_star_ss = np.percentile(SmoothedField, FieldConstPercThreshold)    

            #---------------------------------------------------------------------------------------

            #Compute the fraction of query points satisfying the joint condition
            ind_gtr_k_dt = np.where((vol[:, k]<ss)&(SmoothedField>delta_star_ss))
            p_gtr_k_dt[i] = len(ind_gtr_k_dt[0])/MaskedQueryPosRad.shape[0]

            #---------------------------------------------------------------------------------------

            #Compute the fraction of query points with smoothed field exceeding
            #the overdensity threshold
            ind_gtr_dt = np.where(SmoothedField>delta_star_ss)
            p_gtr_dt[i] = len(ind_gtr_dt[0])/MaskedQueryPosRad.shape[0]

        #-------------------------------------------------------------------------------------------

        p_gtr_k_dt_list.append(p_gtr_k_dt)
        p_gtr_dt_list.append(p_gtr_dt)

        #-------------------------------------------------------------------------------------------

        if Verbose: print('\t\tdone; 
        time taken for step 2: {:.2e} s.'.format(time.perf_counter()-step_2_start_time))

    #-----------------------------------------------------------------------------------------------

    if Verbose: print('\ntotal time taken: {:.2e} s.'.format(time.perf_counter()-total_start_time))
    
    return p_gtr_k_list, p_gtr_dt_list, p_gtr_k_dt_list

####################################################################################################

#----------------------------------------  END OF PROGRAM!  ----------------------------------------

####################################################################################################