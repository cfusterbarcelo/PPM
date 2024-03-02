import os
from PIL import Image
import torchvision
from torchvision.datasets import STL10
from torchvision import datasets, transforms, utils
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

###########################################################################


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):
    def __init__(self, gate_channels=16):

        super().__init__()

        # Convolution and Max Pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.7)

        # Activation
        self.relu = nn.ReLU()

        # Linear layers
        self.fc1 = nn.Linear(35328, 256)
        self.fc2 = nn.Linear(256, 1)
        # sigmoid layer
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sig(x)

        return x


class CNN_trainer(CNN):
    def __init__(self, gate_channels=16, lr=1e-3, epochs=200):

        super().__init__(gate_channels)

        # TRAINING VARIABLES
        self.lr = lr
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.epochs = epochs

        # CRITERION
        self.criterion = nn.BCELoss()

        # LOSS EVOLUTION
        self.loss_during_training = []
        self.valid_loss_during_training = []

        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def trainloop(self, trainloader, validloader):

        self.train()

        for e in range(int(self.epochs)):

            running_loss = 0.0

            for images, labels in tqdm(trainloader):

                images = images.to(self.device)
                labels = labels.to(self.device).view(-1, 1)

                self.optim.zero_grad()

                pred = self.forward(images)

                loss = self.criterion(pred, labels.type(torch.float32))

                loss.backward()

                self.optim.step()

                running_loss += loss.item()

            self.loss_during_training.append(running_loss / len(trainloader))

            # Validation
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                running_loss = 0.0

                for images, labels in tqdm(validloader):

                    images = images.to(self.device)
                    labels = labels.to(self.device).view(-1, 1)

                    pred = self.forward(images)

                    loss = self.criterion(pred, labels.type(torch.float32))

                    running_loss += loss.item()

                self.valid_loss_during_training.append(running_loss / len(validloader))

            print(
                "\nTrain Epoch: {} -> Training Loss: {:.6f}".format(
                    e, self.loss_during_training[-1]
                )
            )
            print(
                "Train Epoch: {} -> Validation Loss: {:.6f}".format(
                    e, self.valid_loss_during_training[-1]
                )
            )

    def eval_performance(self, dataloader):

        loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():

            for images, labels in dataloader:
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)
                probs = self.forward(images)
                loss += self.criterion(
                    probs.view(-1), labels.type(torch.float32).view(-1)
                )

                labels_pred = torch.round(probs)
                equals = labels_pred == labels
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            return accuracy / len(dataloader), loss / len(dataloader)


def crop_my_image(image):
    """Crop the images so only a specific region of interest is shown to my PyTorch model"""
    return transforms.functional.crop(image, left=20, top=13, width=99, height=94)


###########################################################################
class PatientImageDataset(Dataset):
    def __init__(self, data_dir, af_users, non_af_users, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            af_users (list): List of user IDs with AF condition.
            non_af_users (list): List of user IDs without AF condition.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.af_dir = os.path.join(data_dir, "MimicPerformAF_org", "AF")
        self.non_af_dir = os.path.join(data_dir, "MimicPerformAF_org", "Non_AF")
        self.af_users = af_users
        self.non_af_users = non_af_users
        self.transform = transform
        self.images, self.labels = self._load_images()

    def _load_images(self):
        images = []
        labels = []  # 0 for Non-AF, 1 for AF

        # Load Non-AF images
        for img_name in os.listdir(self.non_af_dir):
            user_id = img_name.split("_")[-2]
            if user_id in self.non_af_users:
                img_path = os.path.join(self.non_af_dir, img_name)
                images.append(img_path)
                labels.append(0)

        # Load AF images
        for img_name in os.listdir(self.af_dir):
            user_id = img_name.split("_")[-2]
            if user_id in self.af_users:
                img_path = os.path.join(self.af_dir, img_name)
                images.append(img_path)
                labels.append(1)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)[:3, :, :]

        return image, label


transform = transforms.Compose(
    [
        transforms.Resize((120, 160)),
        transforms.Lambda(crop_my_image),
        transforms.ToTensor(),
    ]
)
# AF Users from 001 to 019
af_users = [
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "019",
]

# Non-AF Users from 001 to 016
non_af_users = [
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
]
# For loop for two cases: case 01 (af from 001 to 010 and non-af from 001 to 008)
# and case 02 (af from 001 to 009 and non-af from 001 to 007)
for i in range(5):
    if i == 0:
        af_users_train = af_users[:10]
        non_af_users_train = non_af_users[:8]
        af_users_test = af_users[10:]
        non_af_users_test = non_af_users[8:]
    else:
        af_users_train = af_users[: (10 - i)]
        non_af_users_train = non_af_users[: (8 - i)]
        af_users_test = af_users[(10 - i) :]
        non_af_users_test = non_af_users[(8 - i) :]

    # Initialize the dataset
    train_dataset = PatientImageDataset(
        data_dir="data",
        af_users=af_users_train,
        non_af_users=non_af_users_train,
        transform=transform,
    )
    test_dataset = PatientImageDataset(
        data_dir="data",
        af_users=af_users_test,
        non_af_users=non_af_users_test,
        transform=transform,
    )

    ###########################################################################

    valid_size = int(0.1 * len(train_dataset))

    part_tr = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) - valid_size, valid_size]
    )[0]

    # Validation partition
    part_val = torch.utils.data.random_split(
        train_dataset, [valid_size, len(train_dataset) - valid_size]
    )[0]

    # Initialize the DataLoader
    train_loader = DataLoader(part_tr, batch_size=256, shuffle=True)
    validloader = DataLoader(part_val, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # print total number of images in train, valid and test
    print(
        f"Train: {len(part_tr)}, Validation: {len(part_val)}, Test: {len(test_dataset)}"
    )

    continue

    # Set cuda enviroment os = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = CNN_trainer(gate_channels=16, lr=1e-3, epochs=50)
    model.trainloop(trainloader=train_loader, validloader=validloader)

    # Evaluate the model
    accuracy, loss = model.eval_performance(test_loader)

    print(f"Accuracy: {accuracy}, Loss: {loss}")

    # Save acc and loss in a txt file with the index of the for
    with open(f"results_{i}.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}, Loss: {loss}")
