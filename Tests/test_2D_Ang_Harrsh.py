import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from kNNpy import kNN_2D_Ang

def make_positions(n, seed=0):
    np.random.seed(seed)
    dec = np.random.uniform(-np.pi/2 + 0.001, np.pi/2 - 0.001, n)
    ra = np.random.uniform(0.001, 2 * np.pi - 0.001, n)
    return np.column_stack((dec, ra))

def make_bins(kList):
    return [np.linspace(0, 0.5, 5) for _ in kList]

# 1. TracerAuto2DA
def test_TracerAuto2DA_valid():
    kList = [1, 2]
    BinsRad = make_bins(kList)
    queries = make_positions(5)
    tracers = make_positions(10, seed=1)
    result = kNN_2D_Ang.TracerAuto2DA(kList, BinsRad, queries, tracers)
    assert isinstance(result, list)
    assert len(result) == len(kList)
    for r in result:
        assert isinstance(r, np.ndarray)

def test_TracerAuto2DA_invalid_declination():
    kList = [1]
    BinsRad = make_bins(kList)
    queries = np.array([[2.0, 0.5]])  # Invalid Declination
    tracers = make_positions(5)
    with pytest.raises(ValueError, match="declination"):
        kNN_2D_Ang.TracerAuto2DA(kList, BinsRad, queries, tracers)

# 2. TracerTracerCross2DA
def test_TracerTracerCross2DA_valid():
    kList = [(1, 1), (2, 2)]
    BinsRad = make_bins(kList)
    queries = make_positions(5)
    tracers_A = make_positions(10)
    tracers_B = make_positions(10, seed=42)
    a, b, c = kNN_2D_Ang.TracerTracerCross2DA(kList, BinsRad, queries, tracers_A, tracers_B)
    assert len(a) == len(kList)
    assert all(isinstance(x, np.ndarray) for x in a)

def test_TracerTracerCross2DA_invalid_query():
    kList = [(1, 1)]
    BinsRad = make_bins(kList)
    queries = np.array([[2.0, 0.5]])  # Invalid Declination
    tracers_A = make_positions(10)
    tracers_B = make_positions(10, seed=42)
    with pytest.raises(ValueError, match="declination"):
        kNN_2D_Ang.TracerTracerCross2DA(kList, BinsRad, queries, tracers_A, tracers_B)

# 3. TracerTracerCross2DA_DataVector
def test_TracerTracerCross2DA_DataVector_valid():
    kList = [(1, 1)]
    BinsRad = make_bins(kList)
    queries = make_positions(5)
    tracers_A_vec = np.array([make_positions(6, seed=i) for i in range(3)])
    tracers_B = make_positions(6)
    a, b, c = kNN_2D_Ang.TracerTracerCross2DA_DataVector(kList, BinsRad, queries, tracers_A_vec, tracers_B)
    assert isinstance(a, list) and isinstance(b, list) and isinstance(c, list)
    assert a[0].shape[0] == 3  # 3 realizations

def test_TracerTracerCross2DA_DataVector_invalid_shape():
    kList = [(1, 1)]
    BinsRad = make_bins(kList)
    queries = make_positions(5)
    tracers_A_vec = np.random.rand(3, 6, 3)  # Invalid Shape
    tracers_B = make_positions(6)
    with pytest.raises(ValueError, match="spatial dimension"):
        kNN_2D_Ang.TracerTracerCross2DA_DataVector(kList, BinsRad, queries, tracers_A_vec, tracers_B)
'''
# 4. TracerFieldCross2DA
def test_TracerFieldCross2DA_valid():
    kList = [1]
    BinsRad = make_bins(kList)
    queries = make_positions(5)
    tracers = make_positions(10)
    field = np.random.rand(12*3*3)
    mask = np.ones_like(field)
    threshold = 75.0
    out = kNN_2D_Ang.TracerFieldCross2DA(kList, BinsRad, queries, tracers, field, mask, threshold)
    assert isinstance(out, tuple) and len(out) == 3

def test_TracerFieldCross2DA_invalid_ra():
    kList = [1]
    BinsRad = make_bins(kList)
    queries = np.array([[0.1, 18.0]])  # Invalid RA
    tracers = make_positions(5)
    field = np.random.rand(12*3*3)
    mask = np.ones_like(field)
    threshold = 75.0
    with pytest.raises(ValueError, match="right ascension"):
        kNN_2D_Ang.TracerFieldCross2DA(kList, BinsRad, queries, tracers, field, mask, threshold)
'''
# 5. TracerFieldCross2DA_DataVector
def test_TracerFieldCross2DA_DataVector_valid():
    kList = [1]
    BinsRad = make_bins(kList)
    queries = make_positions(5)
    tracers_vec = np.array([make_positions(6, seed=i) for i in range(3)])
    field = np.random.rand(12*3*3)
    mask = np.ones_like(field)
    threshold = 75.0
    out = kNN_2D_Ang.TracerFieldCross2DA_DataVector(kList, BinsRad, queries, tracers_vec, field, mask, threshold)
    assert isinstance(out, tuple) and len(out) == 3

def test_TracerFieldCross2DA_DataVector_invalid_dec():
    kList = [1]
    BinsRad = make_bins(kList)
    queries = np.array([[2.0, 0.1]])  # Invalid Declination
    tracers_vec = np.array([make_positions(6, seed=i) for i in range(3)])
    field = np.random.rand(12*3*3)
    mask = np.ones_like(field)
    threshold = 75.0
    with pytest.raises(ValueError, match="declination"):
        kNN_2D_Ang.TracerFieldCross2DA_DataVector(kList, BinsRad, queries, tracers_vec, field, mask, threshold)
        
        

# test_TracerFieldCross2DA_valid()
