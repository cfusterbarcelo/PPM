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
            For this version 1.4
__________________________________________________
## Database used: MIMIC PERform AF Dataset
## Input files: .hea and .dat
## Output files: .pkl
__________________________________________________'''

import pathlib
import os
import wfdb
import heartpy as hp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import building_matrix, normalise, mean_peak_dist

# VARIABLE DECLARATION
sf = 125
# Window size that we want in seconds (20, 30,...)
window_size = 20
# Total duration of the PPG sample, in this case, 20min
ppg_duration = 1200

# Getting PPG files (hea and dat) and declaring other paths
current_dir = pathlib.Path(__file__).resolve()
af_subjects_dir = str(pathlib.Path(current_dir).parents[1] / 'PPG_DDBB/MIMIC_PERform_AF_Dataset/af_subjects/')
non_af_subjects_dir = str(pathlib.Path(current_dir).parents[1] / 'PPG_DDBB/MIMIC_PERform_AF_Dataset/non_af_subjects/')
af_ppm_dir = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/AF/')
non_af_ppm_dir = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/Non_AF/')

# Creating a list of all dat files
dat_records = []
for file in os.listdir(non_af_subjects_dir):
    if file.endswith(".dat"):
        record = file.replace(".dat", "")    
        dat_records.append(record)

# Extract list of PPG peaks for each user
# stored as key:peaks,signal
users_ppg = {}
for record in dat_records:
    sig, fields = wfdb.rdsamp(non_af_subjects_dir+'/'+dat_records[0])
    ppg_signal = sig[:, 0]
    wd, m = hp.process(ppg_signal, sf)
    ppg_info = []
    ppg_info.append(wd['peaklist'])
    ppg_info.append(ppg_signal)
    users_ppg[record] = ppg_info

obtained_users = []
init_window = 0
for key in users_ppg.keys():
    if str(key) in obtained_users:
        continue
    else:
        obtained_users.append(str(key))
        print('User appended ', key)
    
    init_window = 0
    peaks = users_ppg.get(key)[0]
    ppg = users_ppg.get(key)[1]
    window_samp = int((window_size*len(ppg))/ppg_duration)

    norm_ppg = normalise(ppg)
    mean_dist = mean_peak_dist(peaks)
    i=0
    while len(norm_ppg):
        if (init_window >=(len(norm_ppg)-window_samp)): break

        matrix = building_matrix(mean_dist, peaks, norm_ppg, init_window, window_samp)
        norm_matrix = normalise(matrix)
        sns.heatmap(norm_matrix, xticklabels=False, yticklabels=False)
        plt.savefig(non_af_ppm_dir + '/PPM_' + key + '_' + str(i))
        plt.close('all')
        
        init_window += window_samp
        i+=1

