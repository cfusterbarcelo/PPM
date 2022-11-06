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

sf = 125
# Window size that we want in seconds (20, 30,...)
window_size = 20
# Total duration of the PPG sample, in this case, 20min
ppg_duration = 1200
init_window = 0


# == FUNCTIONS
def normalise(signal):
    a, b = -1, 1
    c = b - a
    aux = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    norm_signal = c * aux + a
    return norm_signal

def mean_peak_dist(peaks):
    dist = []
    for i in range(len(peaks)):
        if peaks[i] == peaks[-1]:
            break
        d = peaks[i + 1] - peaks[i]
        if i == 0:
            dist.append(d)
            continue
        if d > np.mean(dist) + np.std(dist) * 2:
            continue
        else:
            dist.append(d)
    return np.mean(dist)

def building_matrix(mean_dist, peaks, norm_ppg, init_window, window_samp):
    all_segments = []
    init_seg = int(0.2 * mean_dist)
    fin_seg = int(1.3 * mean_dist)
    peaks = np.asarray(peaks)
    idx = np.where((peaks > init_window) & (peaks<init_window + window_samp))[0]
    # We need to grab a certain number of peaks depending on the window
    # How many peaks are there in the selected window?
    for peak in peaks[idx]:
        if peak - init_seg < 0:
            segment = norm_ppg[0:peak + fin_seg]
        else:
            segment = norm_ppg[peak-init_seg:peak+fin_seg]
        all_segments.append(segment[:,np.newaxis])
    if all_segments[0].shape[0] < all_segments[1].shape[0]:
        zeros = np.zeros(int(all_segments[1].shape[0])-int(all_segments[0].shape[0]))[:, np.newaxis]
        new_segment = np.concatenate((zeros, all_segments[0]))
        all_segments[0] = new_segment
    # Removing last segment to compute matrix in case sizes do not match
    if len(all_segments[-1])<len(all_segments[-2]):
        all_segments.pop()
    try:
        matrix = np.concatenate(all_segments, 1)
    except ValueError:
        print("MATRIX CANNOT BE COMPUTED!!!!")
        return None
    return matrix.T

# =============

# Getting PPG files (hea and dat)
current_dir = pathlib.Path(__file__).resolve()
af_subjects_dir = str(pathlib.Path(current_dir).parents[1] / 'DDBB/MIMIC_PERform_AF_Dataset/af_subjects/')
non_af_subjects_dir = str(pathlib.Path(current_dir).parents[1] / 'DDBB/MIMIC_PERform_AF_Dataset/non_af_subjects/')



# Creating a list of all dat files
dat_af_records = []
for file in os.listdir(af_subjects_dir):
    if file.endswith(".dat"):
        record = file.replace(".dat", "")    
        dat_af_records.append(record)

# Extract list of PPG peaks for each user
# stored as key:peaks,signal
users_ppg = {}
for record in dat_af_records:
    sig, fields = wfdb.rdsamp(af_subjects_dir+'/'+dat_af_records[0])
    ppg_signal = sig[:, 0]
    wd, m = hp.process(ppg_signal, sf)
    ppg_info = []
    ppg_info.append(wd['peaklist'])
    ppg_info.append(ppg_signal)
    users_ppg[record] = ppg_info

obtained_users = []
for key in users_ppg.keys():
    if str(key) in obtained_users:
        continue
    else:
        obtained_users.append(str(key))
        print('User appended ', key)
    
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
        plt.savefig('ppms/ppm' + str(i))
        plt.close('all')
        init_window += window_samp
        i+=1

