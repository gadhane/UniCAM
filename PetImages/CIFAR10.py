import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import Attention, VanillaKD, RKDLoss

from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models

import os
import sys

import shutup
shutup.please()

ResNetPath = 'models/CIFAR10/CIFAR10_teacher_ResNet50.pt'
teacher_model = models.resnet50(pretrained=False)
num_ftrs = teacher_model.fc.in_features
teacher_model.fc = nn.Linear(num_ftrs, 10)
state = torch.load(ResNetPath)
teacher_model.load_state_dict(state)

student_model = models.resnet18(pretrained=True)
num_ftrs = student_model.fc.in_features
student_model.fc = nn.Linear(num_ftrs, 10)

# This part is where you define your datasets, dataloaders, models and optimizers
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        ),
    ),
    batch_size=16,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "mnist_data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ]
        ),
    ),
    batch_size=16,
    shuffle=True,
)

teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)
student_optimizer = optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)

if __name__=='__main__':
    distiller = Attention(teacher_model, student_model, train_loader, test_loader,
                          teacher_optimizer, student_optimizer, log=True, logdir="CIFAR10Logs/")
    # distiller.train_teacher(epochs=20, plot_losses=False, save_model=True, save_model_pth = 'models/CIFAR10/CIFAR10_teacher_ResNet50.pt')    # Train the teacher network
    distiller.train_student(epochs=25, plot_losses=False, save_model=True, save_model_pth = 'models/CIFAR10/CIFAR10_Att_ResNet18_ResNet50.pt')    # Train the student network

    distiller.evaluate(teacher=False)                                       # Evaluate the student network
    distiller.get_parameters()                                              # A utility function to get the number of
                                                                       # parameters in the  teacher and the student network
