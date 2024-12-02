import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import VanillaKD, Attention

from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

import numpy as np
from sklearn.model_selection import KFold
import os.path

import os
import sys
# import shutup
#
# shutup.please()

import warnings
warnings.filterwarnings("ignore")
# nescalab@uoc.edu
data_dir = './PetImages/'
# data_dir = 'D:/ImageNet/PetImages/PetImages'
k_folds = 5
# ResNetPath = 'models/PetImages_ModelResNet50.pt'
data_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform = data_transforms)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)


if __name__=='__main__':
    batch_size = 32
    print('--------------------------------')
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        # saveTeacher = f'models/Teacher_Resnet50_{fold+1}.pt'
        # teacherperf = f'csv/Teacher_Resnet50_{fold+1}.csv'

        saveStudent = f'models/attKD_Resnet50_ResNet50_{fold+1}.pt'
        studentperf = f'csv/attKD_Resnet50_ResNet50_{fold+1}.csv'

        print(f'FOLD {fold}')
        print('--------------------------------')

        train_dataset = Subset(dataset, train_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = Subset(dataset, val_indices)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        model_path = f'models/Teacher_Resnet50_{fold+1}.pt'
        teacher_model = models.resnet34(pretrained=False)
        num_ftrs = teacher_model.fc.in_features
        teacher_model.fc = nn.Linear(num_ftrs, 2)
        state = torch.load(model_path)
        teacher_model.load_state_dict(state)

        student_model = models.resnet34(pretrained=False)
        num_ftrs = student_model.fc.in_features
        student_model.fc = nn.Linear(num_ftrs, 2)

        teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)
        student_optimizer = optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)

        distiller = Attention(teacher_model, student_model, train_loader, test_loader,
                              teacher_optimizer, student_optimizer)
        # distiller.train_teacher(epochs=20, plot_losses=False, save_model=False,
        #                       save_model_pth = saveTeacher,
        #                       filename = teacherperf)    # Train the teacher network
        distiller.train_student(epochs=25, plot_losses=False, save_model=True,
                                save_model_pth = saveStudent,
                                filename = studentperf)    # Train the student network
        distiller.evaluate(teacher=False)                                       # Evaluate the student network
        distiller.get_parameters()                                              # A utility function to get the number of