''''''
'''_______________________________________________
    Python file to move PPM images
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
## Input files: .jpg
## Output files: .jpg
__________________________________________________'''

import os
import pathlib
import numpy as np
import shutil
import random

train_perc = 0.8

current_dir = pathlib.Path(__file__).resolve()
af_ppm_dir = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/MimicPerformAF_org/AF/')
non_af_ppm_dir = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/MimicPerformAF_org/Non_AF/')
af_list = os.listdir(af_ppm_dir)
non_af_list = os.listdir(non_af_ppm_dir)
random.shuffle(af_list)
random.shuffle(non_af_list)
train_dest = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/MimicPerformAF_v2/Train/')
test_dest = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/MimicPerformAF_v2/Test/')

af_idx = int(np.floor(len(af_list)*train_perc))
non_af_idx = int(np.floor(len(non_af_list)*train_perc))

af_train = af_list[0:af_idx]
af_test = af_list[af_idx +1:]

non_af_train = non_af_list[0:non_af_idx]
non_af_test = non_af_list[non_af_idx +1:]

for file in af_train:
    source = af_ppm_dir + '/' + file
    dest = train_dest+'/AF/'
    shutil.copy(source, dest)

for file in af_test:
    source = af_ppm_dir + '/' + file
    dest = test_dest+'/AF/'
    shutil.copy(source, dest)
