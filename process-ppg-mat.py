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
import matplotlib.pyplot as plt
import heartpy as hp
from scipy.signal import resample

mat_file = '/Users/caterina/Library/CloudStorage/GoogleDrive-cafuster@pa.uc3m.es/La meva unitat/COSEC/PPG/PPG_DDBB/Zenodo-Long-termPPGwithAF/Data/01_PPG_01.mat'

sf = 100

arrays = {}
f = h5py.File(mat_file)
for k, v in f.items():
    arrays[k] = np.array(v)

ppg = arrays.get('PPG_GREEN').flatten()

filtered_ppg = hp.filter_signal(ppg, cutoff= [0.75,3.5], sample_rate=100.0, order=3, filtertype='bandpass')

wd, m = hp.process(filtered_ppg, sample_rate=sf, windowsize=1, bpmmin=0, bpmmax=9999, high_precision=True, clean_rr=False)

#set large figure
plt.figure(figsize=(12,4))

#call plotter
hp.plotter(wd, m)

#display measures computed
for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))