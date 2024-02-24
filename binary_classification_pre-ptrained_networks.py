import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from utilities import crop_tensor, CroppedImageDataset
import timm
from torch import nn

# VARS for each CNN launched to change depending on the DDBB
epochs_num = 100
epochs_str = str(epochs_num) + "e"
num_classes = 1
ddbb = "MimicPerformAF"
batchsize = 16

partition = "Part1"
train_path = "data/" + partition + "/Train/"
test_path = "data/" + partition + "/Test/"
results_path = "data/" + partition + "/results/"
# Create results_path if it does not exist
if not os.path.exists(results_path):
    os.makedirs(results_path)
output_path = "MimicPerformAF_output/Test_10folds/" + partition + "/output/"
# Create output_path if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)
# output_file = output_path + partition + epochs_str + "-outcome.txt"

# orig_stdout = sys.stdout
# f = open(output_file, "w")
# sys.stdout = f

# ==== DATA INITIALIZATION AND PREPROCESSING ====

# Define the transformations: Rescaling and resizing the images
transform = transforms.Compose(
    [
        transforms.Resize((120, 160)),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ]
)

# Assuming the datasets are organized in a directory per class
# Load the dataset
full_dataset = datasets.ImageFolder(root=train_path, transform=transform)

# Splitting the dataset into train and validation
train_size = int(0.9 * len(full_dataset))
validation_size = len(full_dataset) - train_size
train_dataset, validation_dataset = random_split(
    full_dataset, [train_size, validation_size]
)

# Load the test dataset
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

# Define the cropping dimensions
crop_height = (13, 13)  # Crop 13 pixels from the top and 13 pixels from the bottom
crop_width = (20, 41)  # Crop 20 pixels from the left and 41 pixels from the right

# Wrap the original datasets
cropped_train_dataset = CroppedImageDataset(train_dataset, crop_height, crop_width)
cropped_validation_dataset = CroppedImageDataset(
    validation_dataset, crop_height, crop_width
)
cropped_test_dataset = CroppedImageDataset(test_dataset, crop_height, crop_width)

# Create the DataLoaders
train_loader = DataLoader(cropped_train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(cropped_validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(cropped_test_dataset, batch_size=32, shuffle=False)

###### RES NEXT 50 TESTING ######


# Load the pre-trained SE-ResNeXt 50 model
model = timm.create_model("seresnext50_32x4d", pretrained=True)


# Define a transform that matches the model's expectations
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize to match input dimensions
        transforms.CenterCrop(224),  # Crop to the size expected by the models
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Assuming `full_dataset`, `train_dataset`, `validation_dataset`, and `test_dataset` are already defined
# Apply the transform to the datasets
full_dataset.transform = transform
train_dataset.dataset.transform = transform
validation_dataset.dataset.transform = transform
test_dataset.transform = transform

# DataLoaders
batch_size = 32  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modify the model for binary classification
num_features = model.fc.in_features  # Get the number of inputs for the final layer
model.fc = nn.Linear(
    num_features, 2
)  # Replace the final layer for binary classification

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)  # Common choice for fine-tuning

# Training and validation loop
num_epochs = 5  # Number of training epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Validation of the model
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the validation images: {100 * correct / total}%")

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
)


# Assume test_loader is your DataLoader for the test dataset
all_preds = []
all_labels = []
model.eval()  # Make sure model is in eval mode for inference
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds))
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Balanced Accuracy:", balanced_accuracy_score(all_labels, all_preds))


########## The same but for Inception Resnet V2 ##########


# Load the pre-trained Inception-ResNet V2 model
model = timm.create_model("inception_resnet_v2", pretrained=True)

# Define a transform that matches the model's expectations
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize to match Inception input dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Assuming `full_dataset`, `train_dataset`, `validation_dataset`, and `test_dataset` are already defined
# Apply the transform to the datasets
full_dataset.transform = transform
train_dataset.dataset.transform = transform
validation_dataset.dataset.transform = transform
test_dataset.transform = transform

# DataLoaders
batch_size = 32  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modify the model for binary classification
# Inception models have an AuxLogits and fc layer, so you need to adjust both
num_features = (
    model.classif.in_features
)  # Change `.fc` to `.classif` for Inception ResNet V2
model.classif = nn.Linear(
    num_features, 2
)  # Adjust the final layer for binary classification
if hasattr(model, "AuxLogits"):  # In case the Inception model has an auxiliary output
    num_aux_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_aux_features, 2)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)  # Common choice for fine-tuning

# Training and validation loop
num_epochs = 5  # Number of training epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Validation of the model
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the validation images: {100 * correct / total}%")

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
)


# Assume test_loader is your DataLoader for the test dataset
all_preds = []
all_labels = []
model.eval()  # Make sure model is in eval mode for inference
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds))
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Balanced Accuracy:", balanced_accuracy_score(all_labels, all_preds))
