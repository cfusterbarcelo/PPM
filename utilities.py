''''''
'''_______________________________________________
   Utilities file
__________________________________________________
## Author: Caterina Fuster Barceló
## Version: 1.0
## Email: cafuster@pa.uc3m.es
## Status: Development
__________________________________________________

__________________________________________________
            For this version 1.0
__________________________________________________'''

import numpy as np

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