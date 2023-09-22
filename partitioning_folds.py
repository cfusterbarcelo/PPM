''''''
'''_______________________________________________
Python file to create different partitions of users
in order to have 10 different folds into train and
test
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
## Input files: .jpg
## Output files: .jpg
__________________________________________________'''

import os
import shutil
import random
import pathlib
import glob

# Set the path to the original data folder
current_dir = pathlib.Path(__file__).resolve()
data_folder = str(pathlib.Path(current_dir).parents[2] / 'Data/PPM/MimicPerformAF_org/')


# Number of partitions
num_partitions = 10

# Create partitions
for partition_num in range(1, num_partitions + 1):
    # Create a new partition folder
    partition_folder = f"MimicPerformAF_Part{partition_num}"
    os.makedirs(partition_folder, exist_ok=True)
    
    # Initialize lists to hold users for train and test
    train_af_users = []
    train_non_af_users = []
    test_af_users = []
    test_non_af_users = []
    
    # Randomly shuffle the list of users with AF and without AF
    af_users = list(range(1, 20))  # Adjusted for 19 users with AF
    non_af_users = list(range(1, 17))  # Adjusted for 16 users without AF
    
    random.shuffle(af_users)
    random.shuffle(non_af_users)
    
    # Split users into train and test for both AF and Non_AF groups
    train_af_users = af_users[:15]  # 80% of AF users for train
    test_af_users = af_users[15:]   # 20% of AF users for test
    train_non_af_users = non_af_users[:13]  # 80% of Non_AF users for train
    test_non_af_users = non_af_users[13:]   # 20% of Non_AF users for test
    
    # Create 'Train' and 'Test' directories inside the partition folder
    for split in ["Train", "Test"]:
        split_folder = os.path.join(partition_folder, split)
        os.makedirs(split_folder, exist_ok=True)
        
        # Create 'AF' and 'Non_AF' directories inside 'Train' and 'Test'
        for category in ["AF", "Non_AF"]:
            category_folder = os.path.join(split_folder, category)
            os.makedirs(category_folder, exist_ok=True)
            
            # Copy images for the selected users
            if category == "AF":
                users_to_copy = train_af_users if split == "Train" else test_af_users
            else:
                users_to_copy = train_non_af_users if split == "Train" else test_non_af_users
            
            # Print the users_to_copy including which category and split and the partition name
            print(f"{users_to_copy} {category} {split} {partition_folder}")
            
            for user_num in users_to_copy:
                user_images = glob.glob(f"{data_folder}/{category}/PPM_mimic_perform_{category.lower()}_{user_num:03d}_*.png")
                for image_path in user_images:
                    shutil.copy(image_path, category_folder)