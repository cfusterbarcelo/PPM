import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# VARS for each CNN launched to change depending on the DDBB
epochs_num = 100
epochs_str = str(epochs_num)+'e'
num_classes = 1
ddbb = 'MimicPerformAF'
batchsize = 16

partition = 'Part6'
train_path = 'D:/Data/PPM/MimicPerformAF_10fold/' + partition + '/Train/'
test_path = 'D:/Data/PPM/MimicPerformAF_10fold/' + partition + '/Test/'
results_path = 'D:/Models/PPM/MimicPerformAF_10fold/' + partition + '/results/'
# Create results_path if it does not exist
if not os.path.exists(results_path):
    os.makedirs(results_path)
output_path = 'D:/Github/PPM/MimicPerformAF_output/Test_10folds/' + partition + '/output/'
# Create output_path if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_file = output_path + partition + epochs_str + '-outcome.txt'

orig_stdout = sys.stdout
f = open(output_file, 'w')
sys.stdout = f

# ==== DATA INITIALIZATION ====

# Define the transformations: Rescaling and resizing the images
transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),  # Converts to [0, 1] range
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Optional: Normalize images
])

# Assuming the datasets are organized in a directory per class
# Load the dataset
full_dataset = datasets.ImageFolder(root=train_path, transform=transform)

# Splitting the dataset into train and validation
train_size = int(0.9 * len(full_dataset))
validation_size = len(full_dataset) - train_size
train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

# Load the test dataset
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check one batch
for batchX, batchy in train_loader:
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    break  # To only print the first batch

input_shape = [batchX.shape[2], batchX.shape[3], batchX.shape[1]]  # PyTorch uses [C, H, W] format
