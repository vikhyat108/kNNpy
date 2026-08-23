import pytest

import numpy as np

import healpy as hp
from healpy.newvisufunc import projview

from matplotlib import pyplot as plt, ticker as mticker
import matplotlib.colors as colors

import copy

import os
import sys

import warnings

warnings.filterwarnings('ignore')

module_path = os.path.abspath(os.path.join('../../'))           # '../' is needed because the parent directory is one directories upstream of the tutorials directory
if module_path not in sys.path:
    sys.path.append(module_path)

from kNNpy import HelperFunctions as hf                 #some helper functions
from kNNpy import HelperFunctions_2DA as hf_2DA         #2D specifi helper functions
from kNNpy import kNN_2D_Ang                            #the main module
from kNNpy.Data import Datasets                         #helpful for retreiving example datasets

def compute_TracerAuto2DA(k_List, sel_bins, query_pos, ga_pos, ReturnNNdist=False, Verbose=False):

    results = kNN_2D_Ang.TracerAuto2DA(k_List, sel_bins, query_pos, ga_pos, ReturnNNdist, Verbose)
    return results

def test_compute_TracerAuto2DA(benchmark, NSIDE, n_tracer, k_List, rounds, warmup):

    print(NSIDE, n_tracer, k_List, rounds, warmup)

    def setup():
        # Build fresh input for each round. The return value becomes the
        # args/kwargs passed to the measured function, and setup time is NOT counted.

        sim_num = 0
        mask = np.ones(12*NSIDE**2)
        mask[mask!=1] = hp.UNSEEN
        ga_pos_arr, ga_map = Datasets.Sample2DTracersFromQuijoteBox(sim_num=sim_num, tracer_type='Galaxies', mask=mask, N_realisations=1, n_tracers=n_tracer, seed=None, map_NSIDE=64, DataPath='../../kNNpy/Data')
        ga_pos = ga_pos_arr[0]
        n_bar_red = (ga_pos.shape[0]/(4*np.pi))*(12*NSIDE**2/len(np.where(mask!=hp.UNSEEN)[0]))
        bins = np.zeros((len(k_List), 10000))
        for i, k in enumerate(k_List):
            bins[i] = np.deg2rad(np.geomspace(0.05, 10, 10000))
        Theoretical_Uniform_CDFs_test = []
        for i, k in enumerate(k_List):
            Theoretical_Uniform_CDFs_test.append(hf_2DA.PoissonUniformCDFs(2*np.pi*(1-np.cos(bins[i])), n_bar_red, k))
        low_bin = np.zeros(len(k_List)).astype(int)
        high_bin = np.zeros(len(k_List)).astype(int)
        for i, k in enumerate(k_List):
            low_bin[i] = np.searchsorted(Theoretical_Uniform_CDFs_test[i], 0.05)
            high_bin[i] = np.searchsorted(Theoretical_Uniform_CDFs_test[i], 0.95)
        sel_bins = np.zeros((len(k_List), 10))
        for i, k in enumerate(k_List):
            sel_bins[i] = np.geomspace(bins[i][low_bin[i]]*0.95, bins[i][high_bin[i]]*1.05, 10)
        _, query_pos = hf_2DA.create_query_2DA(NSIDE, mask)
        
        return (k_List, sel_bins, query_pos, ga_pos, False, False), {}

    result = benchmark.pedantic(
        compute_TracerAuto2DA,
        setup=setup,
        rounds=rounds,          # number of measured rounds
        iterations=1,           # calls per round (the per-call time is rounds-averaged)
        warmup_rounds=warmup,   # untimed rounds to warm caches/JIT before measuring
    )

    assert len(result) == len(k_List)