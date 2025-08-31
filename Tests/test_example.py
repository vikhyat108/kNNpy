import numpy as np

import os
import sys

#The module that needs to be tested

#Necessary for relative imports (see https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im)
module_path = os.path.abspath(os.path.join('../'))           # '../../' is needed because the parent directory is two directories upstream of this test directory
if module_path not in sys.path:
    sys.path.append(module_path)

from kNNpy import HelperFunctions_2DA as hf

def test_bl_th():

    l = 0
    ss = np.pi/2

    bl = hf.bl_th(l, ss)
    
    assert bl==(1-np.cos(ss))/(4*np.pi*(1-np.cos(ss)))