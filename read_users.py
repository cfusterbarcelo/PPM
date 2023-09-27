import os
import pathlib

# Initialize the paths
current_dir = pathlib.Path(__file__).resolve()
partition_nums = range(1, 11)

for partition_num in partition_nums:
    train_dir = str(pathlib.Path(current_dir).parents[2] / f'Data/PPM/MimicPerformAF_10fold/Part{partition_num}/Train/')
    test_dir = str(pathlib.Path(current_dir).parents[2] / f'Data/PPM/MimicPerformAF_10fold/Part{partition_num}/Test/')

    # Create empty lists to store users
    af_train_users = []
    non_af_train_users = []
    af_test_users = []
    non_af_test_users = []

    # Function to extract user from file name
    def extract_af_user(filename):
        return int(filename.split('_')[4])
    def extract_non_af_user(filename):
        return int(filename.split('_')[5])

    # Process AF Train folder
    for root, _, files in os.walk(os.path.join(train_dir, 'AF')):
        for file in files:
            user = extract_af_user(file)
            af_train_users.append(user)

    # Process Non_AF Train folder
    for root, _, files in os.walk(os.path.join(train_dir, 'Non_AF')):
        for file in files:
            user = extract_non_af_user(file)
            non_af_train_users.append(user)

    # Process AF Test folder
    for root, _, files in os.walk(os.path.join(test_dir, 'AF')):
        for file in files:
            user = extract_af_user(file)
            af_test_users.append(user)

    # Process Non_AF Test folder
    for root, _, files in os.walk(os.path.join(test_dir, 'Non_AF')):
        for file in files:
            user = extract_non_af_user(file)
            non_af_test_users.append(user)

    # Print user lists for the current partition
    print(f'Partition {partition_num}')
    print(f'AF train users: {sorted(list(set(af_train_users)))}')
    print(f'Non AF train users: {sorted(list(set(non_af_train_users)))}')
    print(f'AF test users: {sorted(list(set(af_test_users)))}')
    print(f'Non AF test users: {sorted(list(set(non_af_test_users)))}')
    print('\n')

    # Find intersecting users for AF Train and Test
    af_train_test_intersection = list(set(af_train_users) & set(af_test_users))

    # Find intersecting users for Non AF Train and Test
    non_af_train_test_intersection = list(set(non_af_train_users) & set(non_af_test_users))

    # Print partition number and results
    print(f'Partition {partition_num}')
    if af_train_test_intersection:
        print(f'Repeated users in AF Train and Test: {af_train_test_intersection}')
    else:
        print('No repeated users in AF Train and Test')
    
    if non_af_train_test_intersection:
        print(f'Repeated users in Non AF Train and Test: {non_af_train_test_intersection}')
    else:
        print('No repeated users in Non AF Train and Test')

    print('\n')






