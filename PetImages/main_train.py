import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import VanillaKD, Attention

from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models

import os.path

import os
import sys
import shutup

shutup.please()

import warnings
warnings.filterwarnings("ignore")
# nescalab@uoc.edu
data_dir = 'PetImages'
# data_dir = './PetImages/'
# model_path = 'models/Vanilla_Resnet50_ResNet101.pt'
batch_size = 32


teacher_model = models.resnet18(pretrained=True)
num_ftrs = teacher_model.fc.in_features
teacher_model.fc = nn.Linear(num_ftrs, 2)
# state = torch.load(model_path)
# teacher_model.load_state_dict(state)

student_model = models.resnet18(pretrained=True)
num_ftrs = student_model.fc.in_features
student_model.fc = nn.Linear(num_ftrs, 2)

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
transform_train = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test =  transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])

pin_memory = True
torch.manual_seed(5)

trainset = datasets.ImageFolder(os.path.join(data_dir,'train'), transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1,
                                           pin_memory=pin_memory)

testset = datasets.ImageFolder(os.path.join(data_dir,'test'), transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1,
                                           pin_memory=pin_memory)

# teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)
teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)
student_optimizer = optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)
# Now, this is where KD_Lib comes into the picture

if __name__=='__main__':
    saveStudent = f'models/Vanilla_Resnet18.pt'
    studentperf = f'csv/Vanilla_Resnet18.csv'
    distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader,
                          teacher_optimizer, student_optimizer, logdir='PetImageLogs/VanillaResNet18ResNet18')
    distiller.train_teacher(epochs=20, plot_losses=False, save_model=True,
                          save_model_pth = 'models/teacher_ResNet18.pt',
                          filename = 'csv/Vanilla_Resnet18.csv')    # Train the teacher network
    distiller.train_student(epochs=25, plot_losses=False, save_model=True,
                            save_model_pth=saveStudent,
                            filename=studentperf)   # Train the student network
