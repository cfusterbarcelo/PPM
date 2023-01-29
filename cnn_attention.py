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


class ChannelPool(nn.Module):
    def forward(self, x):
        out = torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )
        return out


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):

        super(ChannelGate, self).__init__()

        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )

    def forward(self, x):

        channel_att_sum = None

        avg_pool = F.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        channel_att_avg = self.mlp(avg_pool)

        max_pool = F.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        channel_att_max = self.mlp(max_pool)

        channel_att = torch.sigmoid(channel_att_avg + channel_att_max)

        scale = (
            channel_att.unsqueeze(2).unsqueeze(3).expand_as(x)
        )  # mask for the three channels

        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, no_spatial=False):
        super(CBAM, self).__init__()

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.no_spatial = no_spatial

        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):

        x_out = self.ChannelGate(x)

        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)

        return x_out


class CNN(nn.Module):
    def __init__(self, gate_channels=16):

        super().__init__()

        # Convolution and Max Pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.7)

        # Attention
        self.attn1 = CBAM(gate_channels=gate_channels)

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

        outattn = self.attn1(x)

        x = outattn.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sig(x)

        return x, outattn


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

            for images, labels in trainloader:

                images = images.to(self.device)
                labels = labels.to(self.device).view(-1, 1)

                self.optim.zero_grad()

                pred, _ = self.forward(images)

                loss = self.criterion(pred, labels.type(torch.float32))

                loss.backward()

                self.optim.step()

                running_loss += loss.item()

            self.loss_during_training.append(running_loss / len(trainloader))

            # Validation
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                running_loss = 0.0

                for images, labels in validloader:

                    images = images.to(self.device)
                    labels = labels.to(self.device).view(-1, 1)

                    pred, _ = self.forward(images)

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
                probs, _ = self.forward(images)

                labels_pred = torch.round(probs)
                equals = labels_pred == labels
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            return accuracy / len(dataloader)


def crop_my_image(image):
    """Crop the images so only a specific region of interest is shown to my PyTorch model"""
    return transforms.functional.crop(image, left=20, top=13, width=99, height=94)


trainset = datasets.ImageFolder(
    "pytorch_example/dataset/Test07/Train",
    transform=transforms.Compose(
        [
            transforms.Resize((120, 160)),
            transforms.Lambda(crop_my_image),
            transforms.ToTensor(),
        ]
    ),
)
valid_size = int(0.1 * len(trainset))

part_tr = torch.utils.data.random_split(
    trainset, [len(trainset) - valid_size, valid_size]
)[0]
# Use sampler for randomization
training_sampler = torch.utils.data.SubsetRandomSampler(range(len(part_tr)))
# Define the dataloader
trainloader = torch.utils.data.DataLoader(
    part_tr, batch_size=32, sampler=training_sampler
)

# Validation partition
part_val = torch.utils.data.random_split(
    trainset, [valid_size, len(trainset) - valid_size]
)[0]
# Use sampler for randomization
valid_sampler = torch.utils.data.SubsetRandomSampler(range(len(part_val)))
# Define the dataloader
validloader = torch.utils.data.DataLoader(
    part_val, batch_size=32, sampler=valid_sampler
)


test_dataset = datasets.ImageFolder(
    "pytorch_example/dataset/Test07/Test",
    transform=transforms.Compose(
        [
            transforms.Resize((120, 160)),
            transforms.Lambda(crop_my_image),
            transforms.ToTensor(),
        ]
    ),
)

dataloader_test = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True, num_workers=4
)


# Train the model
import os

# Set cuda enviroment os = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = CNN_trainer(gate_channels=16, lr=1e-3, epochs=10)
model.trainloop(trainloader=trainloader, validloader=validloader)

# Evaluate the model
model.eval_performance(dataloader_test)


# Analyse attention weights
img, labels = next(iter(dataloader_test))
img, labels = img.to(model.device), labels.to(model.device)

_, attn = model.forward(img)
# Sort from bigger to lower attn weights by dimension 1 (channel
attn = attn.sort(dim=1, descending=True)[0]
fig, axes = plt.subplots(10, 6, figsize=(5 * 5, 10 * 5))
for i in range(10):
    img_plot = np.transpose(img[i, :, :, :].cpu().detach().numpy(), (1, 2, 0))
    img_plot = img_plot / 2 + 0.5
    axes[i, 0].imshow(img_plot)
    # If label is 1, then is Atrial Fibrillation, if 0, then is Normal
    axes[i, 0].set_title("Label: {}".format(labels[i].item()))
    axes[i, 1].imshow(attn[i, 0, :, :].cpu().detach().numpy())
    axes[i, 1].set_title("Attention Mask Channel 0")
    axes[i, 2].imshow(attn[i, 1, :, :].cpu().detach().numpy())
    axes[i, 2].set_title("Attention Mask Channel 1")
    axes[i, 3].imshow(attn[i, 2, :, :].cpu().detach().numpy())
    axes[i, 3].set_title("Attention Mask Channel 2")
    axes[i, 4].imshow(attn[i, 3, :, :].cpu().detach().numpy())
    axes[i, 4].set_title("Attention Mask Channel 3")
    axes[i, 5].imshow(attn[i, 4, :, :].cpu().detach().numpy())
    axes[i, 5].set_title("Attention Mask Channel 4")
