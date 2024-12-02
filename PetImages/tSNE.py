# Import modules
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch import nn

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import shutup
import numpy as np

shutup.please()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'D:/ImageNet/PetImages/PetImages/'
batch_size = 16
#Load Dataset
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
transform_test = {
        'test': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ]),
}

pin_memory = True
torch.manual_seed(5)


testset = datasets.ImageFolder(os.path.join(data_dir,'test'), transform_test['test'])
dataset = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1,
                                           pin_memory=pin_memory)

# Load teacher and student models (assuming they are already trained)
teacher_path = 'models/PetImages_ModelResNet50.pt'
student1_path = 'models/student_vanila_ResNet50_ResNet50.pt'
student2_path = 'models/attKD_Resnet50_ResNet50_2.pt'
student3_path = 'models/featKD_Resnet50_ResNet50_2.pt'

model1 = models.resnet50(pretrained=False)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 2)
state = torch.load(teacher_path)
model1.load_state_dict(state)

model2 = models.resnet50(pretrained=False)
num_ftrs = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs, 2)
state = torch.load(student2_path)
model2.load_state_dict(state)

if __name__=='__main__':
    # Get features and labels from teacher model using dataloader (assuming it is already defined)
    teacher_model, student_model = model1.to(device), model2.to(device)
    teacher_features = []
    student_features = []
    lbl = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataset):
            inputs = inputs.to(device)
            teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            teacher_features.append(teacher_outputs.cpu().numpy())
            student_features.append(student_outputs.cpu().numpy())
            lbl.append(labels.cpu().numpy())

    lbl = np.concatenate(lbl)
    teacher_features = np.concatenate(teacher_features, axis=0)
    student_features = np.concatenate(student_features, axis=0)

    # Use t-SNE to visualize the feature representations
    tsne = TSNE(n_components=2, verbose=1)
    teacher_tsne = tsne.fit_transform(teacher_features)
    student_tsne = tsne.fit_transform(student_features)

    # Plot the t-SNE embeddings
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], c=lbl)
    plt.title('Teacher Model Features')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(student_tsne[:, 0], student_tsne[:, 1], c=lbl)
    plt.title('Student Model Features')
    plt.colorbar()
    plt.show()