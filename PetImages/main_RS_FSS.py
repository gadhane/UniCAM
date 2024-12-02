import os
import sys
import time
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.transforms as transforms

from scipy.stats import gaussian_kde

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torchvision.transforms as tv_transforms
import pandas as pd
import cv2

from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from PIL import Image
import torchvision.transforms.functional as TF
# Now lets see how to evaluate this explanation:
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst
from transformers import BertTokenizer, BertModel

import warnings
warnings.filterwarnings("ignore")


# Define Distance Correlation
class Loss_DC(nn.Module):
    def __init__(self):
        super(Loss_DC, self).__init__()

    def Distance_Correlation(self, latent, control):
        matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim=-1) + 1e-12)
        matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim=-1) + 1e-12)

        matrix_A = matrix_a - torch.mean(matrix_a, dim=0, keepdims=True) - torch.mean(matrix_a, dim=1,
                                        keepdims=True) + torch.mean(matrix_a)
        matrix_B = matrix_b - torch.mean(matrix_b, dim=0, keepdims=True) - torch.mean(matrix_b, dim=1,
                                        keepdims=True) + torch.mean(matrix_b)

        Gamma_XY = torch.sum(matrix_A * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_XX = torch.sum(matrix_A * matrix_A) / (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_YY = torch.sum(matrix_B * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])

        correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r

    def forward(self, latent, control):
        dc_loss = self.Distance_Correlation(latent, control)

        return dc_loss


class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, i, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-i])

    def __call__(self, x):
        return self.feature_extractor(x)

# Define Pearson Correlation
def Peasor_Correlation(latent, control):
    batch_size = latent.shape[0]

    up = (latent - torch.mean(latent, dim=0, keepdims=True)) * (control - torch.mean(control, dim=0, keepdims=True))
    up = torch.sum(up) / batch_size

    down = torch.sum((latent - torch.mean(latent, dim=0, keepdims=True)) ** 2) * torch.sum(
        (control - torch.mean(control, dim=0, keepdims=True)) ** 2)
    down = down / (batch_size ** 2)

    return up / torch.sqrt(down)

def P_Distance_Matrix(latent):
    n = latent.shape[0]
    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1)  + 1e-18)
    matrix_A = matrix_a - torch.sum(matrix_a, dim = 0, keepdims= True)/(n-2) - torch.sum(matrix_a, dim = 1, keepdims= True)/(n-2) \
               + torch.sum(matrix_a)/((n-1)*(n-2))

    diag_A = torch.diag(torch.diag(matrix_A))
    matrix_A = matrix_A - diag_A
    return matrix_A


def bracket_op(matrix_A, matrix_B):
    n = matrix_A.shape[0]
    return torch.sum(matrix_A * matrix_B)/(n*(n-3))


def P_removal(matrix_A, matrix_C):
    result = matrix_A - bracket_op(matrix_A, matrix_C) / bracket_op(matrix_C, matrix_C) * matrix_C
    return result

def Correlation(matrix_A, matrix_B):
    Gamma_XY = bracket_op(matrix_A, matrix_B)
    Gamma_XX = bracket_op(matrix_A, matrix_A)
    Gamma_YY = bracket_op(matrix_B, matrix_B)

    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-18)

    return correlation_r


def P_DC(latent_A, latent_B, ground_truth):
    matrix_A = P_Distance_Matrix(latent_A)
    matrix_B = P_Distance_Matrix(latent_B)
    matrix_GT = P_Distance_Matrix(ground_truth)

    matrix_A_B = P_removal(matrix_A, matrix_B)

    cr = Correlation(matrix_A_B, matrix_GT)

    return cr


def New_DC(latent_A, ground_truth):
    matrix_A = P_Distance_Matrix(latent_A)
    matrix_GT = P_Distance_Matrix(ground_truth)
    cr = Correlation(matrix_A, matrix_GT)

    return cr

similarity = Loss_DC()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'PetImages'

# data_dir = 'PetImages/'
batch_size = 4
# Data Standatdization
normalize = tv_transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Data Transformation
transform_test = {
    'cam': tv_transforms.Compose([
        tv_transforms.Resize([224, 224]),
        tv_transforms.ToTensor(),
        #normalize,
    ]),
}
# Loading Test set
pin_memory = True
torch.manual_seed(5)
testset = datasets.ImageFolder(os.path.join(data_dir,'cam'), transform_test['cam'])
testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size, num_workers=1,pin_memory=pin_memory)
classes = ('Cat','Dog')

teacher_path = 'models/teacher_ResNet18.pt'


model1 = models.resnet18(pretrained=False)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 2)
state = torch.load(teacher_path)
model1.load_state_dict(state)

cor = Loss_DC()
if __name__ == '__main__':
    model1 = model1.to(device)
    model1.eval()
   
    model_list = [model1]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embed_model = BertModel.from_pretrained("bert-base-uncased")
    embed_model.eval()

    class_embedding = []
    for cifar_class in classes:
        encoded_input = tokenizer(cifar_class.lower(), return_tensors='pt')
        output = embed_model(**encoded_input)
        class_embedding.append(output[1])
    class_embedding = torch.cat(class_embedding).to(device)

    #Compare teacher with the ground truth
    total_pdcgt11, total_pdcgt21, total_pdcgt31, total_pdcgt41 = 0, 0, 0, 0
    # with torch.no_grad():
    for batch_idx, (inputs, label) in enumerate(testloader):
        images = inputs
        cls = [z.item() for z in label]

        if batch_idx % 10 == 0:
            print(f'At {batch_idx}')

        if len(inputs)>3:
            outputs1, outputs2, outputs3, outputs4 = [], [], [], []
            for i, img in enumerate(images):
                # print(img.shape)
                img = np.array(TF.to_pil_image(img))
                # # img = img.numpy().transpose((1, 2, 0))
                # print(img.shape)
                # print(img)
                # mean = np.array([0.485, 0.456, 0.406])
                # std = np.array([0.229, 0.224, 0.225])
                # img = std * img + mean
                # img = np.clip(img, 0, 1)
                img = np.float32(img) / 255
                input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
                targets = [ClassifierOutputSoftmaxTarget(cls[i])]

                for m, model in enumerate(model_list):
                    # target_layers = [model.layer4, model.layer3, model.layer2, model.layer1]
                    target_layers = [model.layer4]
                    with HiResCAM(model=model, target_layers=target_layers) as cam:
                        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

                    cam_metric = ROADLeastRelevantFirst(percentile=20)
                    _, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
                    visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
                    visualization = deprocess_image(visualization)
                    # import matplotlib.pyplot as plt
                    #
                    # plt.imshow(visualization)
                    # plt.show()
                    img1 = Image.fromarray(visualization)

                    x = TF.to_tensor(img1)
                    x.unsqueeze_(0)
                    if m == 0:
                        if len(outputs1) == 0:
                            outputs1 = torch.cat((x, x),0)
                        else:
                            outputs1 = torch.cat((outputs1, x), 0)

            # Remove the first repeated tensor
            outputs1 = outputs1[1:, :, :, :]


            # Put to device
            outputs1 = torch.Tensor(outputs1).to(device)
            label = label.to(device)

            # Extract Features
            #1. Teacher Features
            features11 = ResnetFeatureExtractor(2, model1)(outputs1)
            features21 = ResnetFeatureExtractor(3, model1)(outputs1)
            features31 = ResnetFeatureExtractor(4, model1)(outputs1)
            features41 = ResnetFeatureExtractor(5, model1)(outputs1)

            features11 = features11.view([features11.shape[0], -1])
            features21 = features21.view([features21.shape[0], -1])
            features31 = features31.view([features31.shape[0], -1])
            features41 = features41.view([features41.shape[0], -1])

            

            # Get the class embedding from pre-trained BERT model
            ground_truth = class_embedding[label]
            # Comparing information in each layer with the ground truth
            # ========================================================
            # Comparing with Ground Truth
            # 1. Teacher
            pdcor1Gt1 = New_DC(features11, ground_truth)
            total_pdcgt11 += pdcor1Gt1.detach()
            pdcor1Gt2 = New_DC(features21, ground_truth)
            total_pdcgt21 += pdcor1Gt2.detach()
            pdcor1Gt3 = New_DC(features31, ground_truth)
            total_pdcgt31 += pdcor1Gt3.detach()
            pdcor1Gt4 = New_DC(features41, ground_truth)
            total_pdcgt41 += pdcor1Gt4.detach()

    from contextlib import redirect_stdout

    with open('distance/NewCalculateVisualConcepts.txt', 'w') as f:
        with redirect_stdout(f):
            
            print(f'Comparing Basemodel layers with ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcgt11 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcgt21 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcgt31 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcgt41 / len(testloader)
            print(model1_info_rem3)