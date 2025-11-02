import os
import sys
import pytest
import numpy as np

#The module that needs to be tested

#Necessary for relative imports (see https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im)
module_path = os.path.abspath(os.path.join('../'))           # '../' is needed because the parent directory is one directory upstream of this test directory
if module_path not in sys.path:
    sys.path.append(module_path)

from kNNpy import kNN_3D
from kNNpy.HelperFunctions import create_query_3D

#--------------------------------------------------------------------------------------------------------------------------

#INPUT VALIDATION TESTS

#--------------------------------------------------------------------------------------------------------------------------

QPos_invalid_1 = create_query_3D('grid', 5, 1000.) + 150.          # Some points invalid (greater than boxsize)
QPos_invalid_2 = create_query_3D('grid', 5, 1000.) - 10.           # Some points invalid (less than 0)
QPos_invalid_3 = np.random.uniform(0, 1000., size=(5**3, 2))       # Invalid shape (only 2 columns instead of 3)
TPos_invalid_1 = np.random.uniform(0, 1200., size=(100, 3))        # Some points invalid (greater than boxsize)
TPos_invalid_2 = np.random.uniform(-50., 1000., size=(100, 3))     # Some points invalid (less than 0)
TPos_invalid_3 = np.random.uniform(0, 1000., size=(100, 4))        # Invalid shape (4 columns instead of 3)

#--------------------------------------------------------------------------------------------------------------------------

arguments = "QueryPos, TracerPos"
parameters = [(QPos_invalid_1, TPos_invalid_1), (QPos_invalid_2, TPos_invalid_2), (QPos_invalid_3, TPos_invalid_3)]
@pytest.mark.parametrize(arguments, parameters)

#Tests the input validation of the TracerAuto3D function in the kNN_3D module. Specifically, it checks whether the function raises a ValueError when provided with tracer or query positions that are outside the valid range [0, boxsize), or not of the correct shape.
def test_TracerAuto3D_input_validation(QueryPos, TracerPos):
    
    #Define other necessary parameters
    kList = [1]
    BinsRad = [np.linspace(10, 45, 3)]
    
    #Do the test
    #Check if the function raises a ValueError if the parameters provided are outside the expected range
    with pytest.raises(ValueError):
        output = kNN_3D.TracerAuto3D(1000., kList, BinsRad, QueryPos, TracerPos)

#--------------------------------------------------------------------------------------------------------------------------

kA_kB_list = [(1, 1), (2, 2)]
kA_kB_list_invalid = [(1, 2), (3, 4), (5, 6)]

arguments = "kA_kB_list, QueryPos, TracerPos_A, TracerPos_B"
parameters = [(kA_kB_list, QPos_invalid_1, TPos_invalid_1, TPos_invalid_2), (kA_kB_list, QPos_invalid_2, TPos_invalid_2, TPos_invalid_1), (kA_kB_list, QPos_invalid_3, TPos_invalid_3, TPos_invalid_3), (kA_kB_list_invalid, create_query_3D('grid', 5, 1000.), np.random.uniform(0, 1000., size=(100, 3)), np.random.uniform(0, 1000., size=(100, 3)))]
@pytest.mark.parametrize(arguments, parameters)

#Tests the input validation of the TracerTracerCross3D function in the kNN_3D module. Specifically, it checks whether the function raises a ValueError when provided with tracer or query positions that are outside the valid range [0, boxsize), or not of the correct shape, or when the kA_kB_list is inconsistent in shape with the distance bins.
def test_TracerTracerCross3D_input_validation(kA_kB_list, QueryPos, TracerPos_A, TracerPos_B):
    
    #Define other necessary parameters
    BinsRad = [np.linspace(10, 45, 3), np.linspace(10, 45, 3)]
    
    #Do the test
    #Check if the function raises a ValueError if the parameters provided are outside the expected range
    with pytest.raises(ValueError):
        output = kNN_3D.TracerTracerCross3D(1000., kA_kB_list, BinsRad, QueryPos, TracerPos_A, TracerPos_B)

#--------------------------------------------------------------------------------------------------------------------------

arguments = "QueryPos, TracerPos"
parameters = [(QPos_invalid_1, TPos_invalid_1), (QPos_invalid_2, TPos_invalid_2), (QPos_invalid_3, TPos_invalid_3)]
@pytest.mark.parametrize(arguments, parameters)

#Tests the input validation of the TracerAuto3D function in the kNN_3D module. Specifically, it checks whether the function raises a ValueError when provided with tracer or query positions that are outside the valid range [0, boxsize), or not of the correct shape.
def test_TracerFieldCross3D_input_validation(QueryPos, TracerPos):
    
    #Define other necessary parameters
    kList = [1, 2]
    BinsRad = [np.linspace(10, 45, 3), np.linspace(10, 45, 3)]
    Field3D = np.random.uniform(0, 1, size=(5, 5, 5))
    
    #Do the test
    #Check if the function raises a ValueError if the parameters provided are outside the expected range
    with pytest.raises(ValueError):
        output = kNN_3D.TracerFieldCross3D(kList, BinsRad, 1000., QueryPos, TracerPos, Field3D, 75.0)

#--------------------------------------------------------------------------------------------------------------------------

#Auto CDF saturation test

def test_TracerAuto3D_CDF_saturation():
    
    #Define parameters
    boxsize = 100.0
    kList = [1, 4, 8]
    BinsRad1 = [np.linspace(110, 200, 10)]*3
    BinsRad2 = [np.linspace(1e-7, 1e-5, 10)]*3
    QueryPos = create_query_3D('grid', 100, boxsize)
    TracerPos = np.random.uniform(0, boxsize, size=(100, 3))
    
    #Get the output
    output1 = kNN_3D.TracerAuto3D(boxsize, kList, BinsRad1, QueryPos, TracerPos)
    output2 = kNN_3D.TracerAuto3D(boxsize, kList, BinsRad2, QueryPos, TracerPos)
    
    #Check if the CDF has saturated to 1
    for k_ind, k in enumerate(kList):
        assert np.all(np.isclose(output1[0][k_ind], np.ones_like(output1[0][k_ind]), atol=1e-5)) and np.all(np.isclose(output2[0][k_ind], np.zeros_like(output2[0][k_ind]), atol=1e-5))

#--------------------------------------------------------------------------------------------------------------------------