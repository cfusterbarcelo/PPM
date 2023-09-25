''''''
'''_______________________________________________
    Python file to move PPM images to create
    different folds for same database
__________________________________________________
## Author: Caterina Fuster Barceló
## Version: 1.0
## Email: cafuster@pa.uc3m.es
## Status: Development
__________________________________________________

__________________________________________________
            For this version 2.0
__________________________________________________
## Database used: MIMIC PERform AF Dataset
## Input files: .png
## Output files: .png
__________________________________________________'''

import os
import shutil
import random
import pathlib

# Set the source directories
current_dir = pathlib.Path(__file__).resolve()
af_ppm_dir = str(pathlib.Path(current_dir).parents[2] / 'Data/PPM/MimicPerformAF_org/AF/')
non_af_ppm_dir = str(pathlib.Path(current_dir).parents[2] / 'Data/PPM/MimicPerformAF_org/Non_AF/')

# Define the number of partitions and users for each
num_partitions = 10
num_af_users = 19
num_non_af_users = 16

# Define the destination directory for partitions
for partition_num in range(1, num_partitions + 1):
    print(f'Creating partition {partition_num}............................................................')
    train_dest = str(pathlib.Path(current_dir).parents[2] / f'Data/PPM/MimicPerformAF_10fold/Part{partition_num}/Train/')
    test_dest = str(pathlib.Path(current_dir).parents[2] / f'Data/PPM/MimicPerformAF_10fold/Part{partition_num}/Test/')

    # Check if partition folders already exist with content
    if os.path.exists(train_dest) and os.listdir(train_dest):
        print(f'Partition {partition_num} already exists with content, skipping.')
        continue

    # Create the directories if they don't exist
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    # Randomly shuffle the user IDs
    af_user_ids = list(range(1, num_af_users + 1))
    non_af_user_ids = list(range(1, num_non_af_users + 1))
    random.shuffle(af_user_ids)
    random.shuffle(non_af_user_ids)

    # Calculate the number of users for training and testing
    num_af_train_users = 15
    num_non_af_train_users = 13
    num_af_test_users = num_af_users - num_af_train_users
    num_non_af_test_users = num_non_af_users - num_non_af_train_users

    # Split users into training and testing
    af_train_users = af_user_ids[:num_af_train_users]
    non_af_train_users = non_af_user_ids[:num_non_af_train_users]
    af_test_users = af_user_ids[num_af_train_users:]
    non_af_test_users = non_af_user_ids[num_non_af_train_users:]

    # Copy all files for each user to the appropriate partitions
    for user_id in af_train_users:
        src_dir = os.path.join(af_ppm_dir, '')
        dest_dir = os.path.join(train_dest, 'AF')
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(src_dir):
            print('AF train users: ', af_train_users)
            # Iterate through all files in the source directory and copy the ones matching user_id
            for file_name in os.listdir(src_dir):
                if file_name.startswith(f'PPM_mimic_perform_af_{user_id:03d}'):
                    src_path = os.path.join(src_dir, file_name)
                    dest_path = os.path.join(dest_dir, file_name)
                    shutil.copy(src_path, dest_path)
        else:
            print('Path not found: ', src_dir)

    for user_id in non_af_train_users:
        src_dir = os.path.join(non_af_ppm_dir, '')
        dest_dir = os.path.join(train_dest, 'Non_AF')
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(src_dir):
            print('Non AF train users: ', non_af_train_users)
            # Iterate through all files in the source directory and copy the ones matching user_id
            for file_name in os.listdir(src_dir):
                if file_name.startswith(f'PPM_mimic_perform_non_af_{user_id:03d}'):
                    src_path = os.path.join(src_dir, file_name)
                    dest_path = os.path.join(dest_dir, file_name)
                    shutil.copy(src_path, dest_path)
                    
        else:
            print('Path not found: ', src_dir)
        

    for user_id in af_test_users:
        src_dir = os.path.join(af_ppm_dir, '')
        dest_dir = os.path.join(test_dest, 'AF')
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(src_dir):
            print('AF test users: ', af_test_users)
            # Iterate through all files in the source directory and copy the ones matching user_id
            for file_name in os.listdir(src_dir):
                if file_name.startswith(f'PPM_mimic_perform_af_{user_id:03d}'):
                    src_path = os.path.join(src_dir, file_name)
                    dest_path = os.path.join(dest_dir, file_name)
                    shutil.copy(src_path, dest_path)
        else:
            print('Path not found: ', src_dir)

    for user_id in non_af_test_users:
        src_dir = os.path.join(non_af_ppm_dir, '')
        dest_dir = os.path.join(test_dest, 'Non_AF')
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(src_dir):
            print('Non AF test users: ', non_af_test_users)
            # Iterate through all files in the source directory and copy the ones matching user_id
            for file_name in os.listdir(src_dir):
                if file_name.startswith(f'PPM_mimic_perform_non_af_{user_id:03d}'):
                    src_path = os.path.join(src_dir, file_name)
                    dest_path = os.path.join(dest_dir, file_name)
                    shutil.copy(src_path, dest_path)
        else:
            print('Path not found: ', src_dir)

print(f'Partition {partition_num} completed.')

print('All partitions created successfully.')

