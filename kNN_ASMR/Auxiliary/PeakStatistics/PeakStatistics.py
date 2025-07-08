import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import scipy
import healpy as hp

def PeakCurves(DensityFields=[], Nreals=10, MaxThreshold=16, Nthresh=101, Type=0, Plot=1, LogScale=1, CosmoLabels=['null']):
    '''
    Gives the peak curves for the given cosmologies' square (over)density fields.
    
    Parameters
    ----------
    DensityFields: float array of shape ''(nCosmo, NofRealizations, XdfDim, YdfDim)''
        4D Array of the 2D (over)density fields of the various cosmologies to be compared. The array should of shape (nCosmo, NofRealizations, XdfDim, YdfDim) 
        where 'nCosmo' is the number of cosmologies to be compared, 'NofRealizations' is the number of realizations of each input cosmology 
        (NofRealizations >= Nreals), and 'XdfDim' and 'YdfDim' are the dimensions of the 2D density fields in pixels.
        Example: np.array(DensityFields).shape = (3, 10, 512, 512) - 3 cosmologies containing 10 realisations each of (512x512) pixel 2D density fields.
    Nreals: int
        Number of realisations of the density fields to be used for the peak curves' statistics. Naturally, cannot be larger than the inherent number 
        of realisations of the 2D density fields contained within the (.npy) data files input, i.e. in the above example, Nreals <=10. Any non-int values 
        are type-cast to int.
    MaxThreshold: float
        Maximum overdensity threshold for which the peak values are to be plotted.
    Nthresh: int
        Number of overdensity threshold (X-axis) values to be computed in the closed interval [-1,MaxThreshold]. Any non-int values are type-cast to int.
    Type: int: 0 or 1, optional
        Type of peak curve to be plotted - 0 for raw peak curve plot; 1 for peak curves normalized by the first input cosmology's peak curve.
    Plot: int: 0 or 1, optional
        1 to plot the output and return results, 0 to skip plotting and only return results. Any values other than 1 will skip plotting.
    LogScale: int: 0 or 1, optional
        1 to plot the peak curves on log Y-axis, 0 to not use log Y-axis. Any values other than 1 will not output the log scale.
    CosmoLabels: str array of shape ''(nCosmo)''
        List of the names/labels to be assigned to the respective input cosmologies. Must have length equal to the number of cosmologies input ('nCosmo').
        Example: ['CDM', 'SIDM', 'FDM']
            
    Returns
    -------
    thresh, tmean, tstddev: 3 numpy arrays containing the threshold values, mean peak values and std. dev. of the peak values
        For (Type = 0): Peak curve plot of the various input comologies' density fields.
        For (Type = 1): Peak curve plot of the various input comologies' density fields normalized by the first input cosmology's density field.
        In both cases, the threshold values array (X-axis, 1D) 'thresh', the peaks array (Y-axis, 2D) 'tmean' containing the number of peaks corresponding to 
        the thresholds array for each input cosmology and their corresponding standard deviations (error, 2D) 'tstddev' are also returned.

    Raises
    ------
    ValueError
        If 'Type' is not 0 or 1.
    ValueError
        If 'Nreals' is lesser than 1.
    ValueError
        If 'MaxThreshold' is lesser than or equal to (-1).
    ValueError
        If the plot needs to be output and the number of labels ('CosmoLabels') is not equal to the number of input cosmologies.
    '''
    if((Type!=0) and (Type!=1)):
        print("ERROR: Incorrect output type requested. Please use either 0 or 1 for variable 'Type'.")
    elif(Nreals<=0):
        print("ERROR: Invalid number of realizations entered. Please enter a positive integer value.")
    elif(MaxThreshold<=-1):
        print("ERROR: Please enter a maximum overdensity threshold value >(-1).")
    elif((Plot==1) and (len(CosmoLabels)!=len(DensityFields))):
        print("ERROR: Please enter the labels for each input cosmology.")
    else:
        Nreals = int(Nreals)
        Nthresh = int(Nthresh)
        b = np.stack(DensityFields)
        tmean = double([[0]*Nthresh]*b.shape[0])
        tstddev = double([[0]*Nthresh]*b.shape[0])
        n_cosmos, n_realizations, height, width = b.shape
        peaksarr = np.zeros((n_cosmos, Nthresh, Nreals), dtype=np.float64)
        thresh = np.linspace(-1, MaxThreshold, Nthresh)
        for cosmo in range(n_cosmos):
            for z in range(Nreals):
                bz = b[cosmo, z]
                local_max = (scipy.ndimage.maximum_filter(bz, size=3, mode='wrap')==bz)
                for t_idx, t in enumerate(thresh):
                    threshold_mask = (bz>=t)
                    peaks_mask = (local_max & threshold_mask)
                    peaksarr[cosmo, t_idx, z] = np.sum(peaks_mask)
            print('%d%% Done.'%(((double(cosmo)*double(Nreals))+double(z)+1.0)/(0.01*double(Nreals)*b.shape[0])))
        if(Type==0):
            for cosmo in range(0,b.shape[0],1):
                for t in range(0,Nthresh,1):
                    tmean[cosmo,t]=np.mean(peaksarr[cosmo,t])
                    tstddev[cosmo,t]=np.std(peaksarr[cosmo,t])
                if(Plot==1):
                    plt.plot(thresh,tmean[cosmo],label=CosmoLabels[cosmo])       
                    plt.fill_between(thresh, tmean[cosmo]-tstddev[cosmo], tmean[cosmo]+tstddev[cosmo], alpha=0.2)
                    plt.title('Peak Curves of Various Cosmologies')
        elif(Type==1):
            for cosmo in range(0,b.shape[0],1):
                for t in range(0,Nthresh,1):
                    tmean[cosmo,t]=np.mean(peaksarr[cosmo,t])/np.mean(peaksarr[0,t])
                    tstddev[cosmo,t]=np.std(peaksarr[cosmo,t])/np.mean(peaksarr[0,t])
                if(Plot==1):
                    plt.plot(thresh,tmean[cosmo],label=CosmoLabels[cosmo])       
                    plt.fill_between(thresh, tmean[cosmo]-tstddev[cosmo], tmean[cosmo]+tstddev[cosmo], alpha=0.2)
                    plt.title('Peak Curves of Various Cosmologies Normalized by ' + CosmoLabels[0])
        if(Plot==1):
            plt.xlabel('Overdensity Threshold')
            plt.ylabel('Number of Peaks')
            if(LogScale==1):
                plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.02,1),loc='upper left',borderaxespad=0)
            plt.show()
        return thresh, tmean, tstddev
    
    
    
def spherical_peaks(denslice,MaxThreshold,Nthresh):
    # Peak Finder Helper Function
    nside = int(np.sqrt(len(denslice)/12))
    nops = 0
    peaks = []
    thresh = []
    neigh = np.copy(hp.pixelfunc.get_all_neighbours(nside,range(len(denslice))))
    for t in linspace(-1,MaxThreshold,Nthresh):
        peax = np.copy(denslice[np.where((denslice[:]>=t)*(denslice[:]>=denslice[neigh[0,:]])*(denslice[:]>=denslice[neigh[1,:]])*(denslice[:]>=denslice[neigh[2,:]])*(denslice[:]>=denslice[neigh[3,:]])*(denslice[:]>=denslice[neigh[4,:]])*(denslice[:]>=denslice[neigh[5,:]])*(denslice[:]>=denslice[neigh[6,:]])*(denslice[:]>=denslice[neigh[7,:]]))])
        nops = len(peax)
        peaks.append(nops)
        thresh.append(t)
    return thresh,peaks

def PeakCurves_Healpix(DensityFields=[], Nreals=10, MaxThreshold=16, Nthresh=101, Type=0, Plot=1, LogScale=1, CosmoLabels=['null']):
    '''
    Gives the peak curves for the given cosmologies' square (over)density fields.
    
    Parameters
    ----------
    DensityFields: float array of shape ''(nCosmo, NofRealizations, nPixels)''
        3D Array of the healpix (over)density fields of the various cosmologies to be compared. The array should of shape (nCosmo, NofRealizations, nPixels) 
        where 'nCosmo' is the number of cosmologies to be compared, 'NofRealizations' is the number of realizations of each input cosmology 
        (NofRealizations >= Nreals), and 'nPixels' is the number of pixels in the healpix projected (over)density field such that [nPixels = 12*res*res], 
        where 'res' is the resolution of the healpix map.
        Example: np.array(DensityFields).shape = (3, 10, 512, 512) - 3 cosmologies containing 10 realisations each of (512x512) pixel 2D density fields.
    Nreals: int
        Number of realisations of the density fields to be used for the peak curves' statistics. Naturally, cannot be larger than the inherent number 
        of realisations of the 2D density fields contained within the (.npy) data files input, i.e. in the above example, Nreals <=10. Any non-int values 
        are type-cast to int.
    MaxThreshold: float
        Maximum overdensity threshold for which the peak values are to be plotted.
    Nthresh: int
        Number of overdensity threshold (X-axis) values to be computed in the closed interval [-1,MaxThreshold]. Any non-int values are type-cast to int.
    Type: int: 0 or 1, optional
        Type of peak curve to be plotted - 0 for raw peak curve plot; 1 for peak curves normalized by the first input cosmology's peak curve.
    Plot: int: 0 or 1, optional
        1 to plot the output and return results, 0 to skip plotting and only return results. Any values other than 1 will skip plotting.
    LogScale: int: 0 or 1, optional
        1 to plot the peak curves on log Y-axis, 0 to not use log Y-axis. Any values other than 1 will not output the log scale.
    CosmoLabels: str array of shape ''(nCosmo)''
        List of the names/labels to be assigned to the respective input cosmologies. Must have length equal to the number of cosmologies input ('nCosmo').
        Example: ['CDM', 'SIDM', 'FDM']
    
    Returns
    -------
    thresh, tmean, tstddev: 3 numpy arrays containing the threshold values, mean peak values and std. dev. of the peak values
        For (Type = 0): Peak curve plot of the various input comologies' density fields.
        For (Type = 1): Peak curve plot of the various input comologies' density fields normalized by the first input cosmology's density field.
        In both cases, the threshold values array (X-axis, 1D) 'thresh', the peaks array (Y-axis, 2D) 'tmean' containing the number of peaks corresponding
        to the thresholds array for each input cosmology and their corresponding standard deviations (error, 2D) 'tstddev' are also returned.

    Raises
    ------
    ValueError
        If 'Type' is not 0 or 1.
    ValueError
        If 'Nreals' is lesser than 1.
    ValueError
        If 'MaxThreshold' is lesser than or equal to (-1).
    ValueError
        If the plot needs to be output and the number of labels ('CosmoLabels') is not equal to the number of input cosmologies.
    '''
    if((Type!=0) and (Type!=1)):
        print("ERROR: Incorrect output type requested. Please use either 0 or 1 for variable 'Type'.")
    elif(Nreals<=0):
        print("ERROR: Invalid number of realizations entered. Please enter a positive integer value.")
    elif(MaxThreshold<=-1):
        print("ERROR: Please enter a maximum overdensity threshold value >(-1).")
    elif((Plot==1) and (len(CosmoLabels)!=len(DensityFields))):
        print("ERROR: Please enter the labels for each input cosmology.")
    else:
        Nreals = int(Nreals)
        Nthresh = int(Nthresh)
        b = np.stack(DensityFields)
        peaksarr = double([[[0]*Nthresh]*Nreals]*b.shape[0])
        tmean = double([[0]*Nthresh]*b.shape[0])
        tstddev = double([[0]*Nthresh]*b.shape[0])
        thresh = double([0]*Nthresh)
        index = 0
        for cosmo in range(0,b.shape[0],1):
            for z in range(0,Nreals,1):
                index += 1
                thresh, peaksarr[cosmo,z] = spherical_peaks(b[cosmo,z],MaxThreshold,Nthresh)
                if(index == b.shape[0]):
                    print('%d%% Done.'%(((double(cosmo)*double(Nreals))+double(z)+1.0)/(0.01*double(Nreals)*b.shape[0])))
                    index = 0
        if(Type == 0):
            for cosmo in range(0,b.shape[0],1):
                tmean[cosmo] = np.mean(peaksarr[cosmo],axis=0)
                tstddev[cosmo] = np.std(peaksarr[cosmo],axis=0)
                if(Plot==1):
                    plt.plot(thresh,tmean[cosmo],label=CosmoLabels[cosmo])       
                    plt.fill_between(thresh, tmean[cosmo]-tstddev[cosmo], tmean[cosmo]+tstddev[cosmo], alpha=0.2)
                    plt.title('Peak Curves of Various Cosmologies')
        elif(Type == 1):
            for cosmo in range(0,b.shape[0],1):
                tmean[cosmo] = np.mean(peaksarr[cosmo],axis=0)/np.mean(peaksarr[0],axis=0)
                tstddev[cosmo] = np.std(peaksarr[cosmo],axis=0)/np.mean(peaksarr[0],axis=0)
                if(Plot==1):
                    plt.plot(thresh,tmean[cosmo],label=CosmoLabels[cosmo])       
                    plt.fill_between(thresh, tmean[cosmo]-tstddev[cosmo], tmean[cosmo]+tstddev[cosmo], alpha=0.2)
                    plt.title('Peak Curves of Various Cosmologies Normalized by ' + CosmoLabels[0])
        if(Plot==1):
            plt.xlabel('Overdensity Threshold')
            plt.ylabel('Number of Peaks')
            if(LogScale == 1):
                plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.02,1),loc='upper left',borderaxespad=0)
            plt.show()
        return thresh, tmean, tstddev