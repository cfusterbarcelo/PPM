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

sf = 125

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

