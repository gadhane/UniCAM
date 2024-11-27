# This code trains a Teacher and Student model using Knowledge Distillation (KD).
# It leverages the kd_lib library and provides flexibility to choose the desired KD type
# for training the Student model.

import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

import copy
import os

from KD_Lib.KD import VanillaKD

import warnings
warnings.filterwarnings("ignore")

# Data Standatdization
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

# Data Transformation
transform_train = {
    'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
}
transform_valid = {
        'valid': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ]),
}

data_dir = 'Images/Data'
batch_size = 16
# Loading Train set
trainset = datasets.ImageFolder(os.path.join(data_dir,'train'), transform_train['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

# Loading Test set
validset = datasets.ImageFolder(os.path.join(data_dir,'valid'), transform_valid['valid'])
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=1)

# Define the Model
teacher = models.resnet50(pretrained=True)
num_ftrs = teacher.fc.in_features
teacher.fc = nn.Linear(num_ftrs, 38)

student = models.resnet50(pretrained=True)
num_ftrs = student.fc.in_features
student.fc = nn.Linear(num_ftrs, 38)

teacher_optimizer = optim.SGD(teacher.parameters(), 0.01)
student_optimizer = optim.SGD(student.parameters(), 0.01)

teacherperformance = f'csv/PlantDiseaseTeacher_Resnet50.csv'
studentperformance = f'csv/PlantDisease_VanillaKD_Resnet50_ResNet50.csv'

if __name__ == '__main__':

    # Now, this is where KD_Lib comes into the picture
    distiller = VanillaKD(teacher, student, trainloader, validloader,
                          teacher_optimizer, student_optimizer)
    distiller.train_teacher(epochs=20, plot_losses=False, save_model=True, filename = teacherperformance)    # Train the teacher network
    distiller.train_student(epochs=20, plot_losses=False, save_model=True, filename = teacherperformance)    # Train the student network
    # distiller.evaluate(teacher=False)                                       # Evaluate the student network
    # distiller.get_parameters()                                              # A utility function to get the number of
