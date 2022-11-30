''''''
'''_______________________________________________
    Python file process the PPG and obtain a pkl
__________________________________________________
## Author: Caterina Fuster Barceló
## Version: 1.0
## Email: cafuster@pa.uc3m.es
## Status: Development
__________________________________________________

__________________________________________________
            For this version 1.0
__________________________________________________
## Database used: MIMIC PERform AF Dataset
## Input files: .mat
## Output files: .png
__________________________________________________'''
import h5py
import numpy as np

mat_file = '/Users/caterina/Documents/Zenodo-Long-termPPGwithAF/Data/01_PPG_01.mat'

arrays = {}
f = h5py.File(mat_file)
for k, v in f.items():
    arrays[k] = np.array(v)