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
batch_size = 5
# Data Standatdization
normalize = tv_transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Data Transformation
transform_test = {
    'test': tv_transforms.Compose([
        tv_transforms.Resize([224, 224]),
        tv_transforms.ToTensor(),
        #normalize,
    ]),
}
# Loading Test set
pin_memory = True
torch.manual_seed(5)
testset = datasets.ImageFolder(os.path.join(data_dir,'test'), transform_test['test'])
testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size, num_workers=1,pin_memory=pin_memory)
classes = ('Cat','Dog')


# teacher_path = 'models/Teacher_Resnet50_ResNet50Nopre.pt'
# student1_path = 'models/student_Feature_ResNet50Nopre_ResNet50.pt'
# # student2_path = 'models/student_Vanilla_Resnet50Nopre_ResNet50.pt'
# student3_path = 'models/student_Att_Resnet50Nopre_ResNet50.pt'

teacher_path = 'models/PetImages_ModelResNet50.pt'
student1_path = 'models/PetImages_FeatureKdResNet50.pt'
student2_path = 'models/student_attKd_ResNet50_ResNet50.pt'
# student3_path = 'models/student_vanila_ResNet50_ResNet50.pt'

model1 = models.resnet50(pretrained=False)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 2)
state = torch.load(teacher_path)
model1.load_state_dict(state)

model2 = models.resnet50(pretrained=False)
num_ftrs = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs, 2)
state = torch.load(student1_path)
model2.load_state_dict(state)

model3 = models.resnet50(pretrained=False)
num_ftrs = model3.fc.in_features
model3.fc = nn.Linear(num_ftrs, 2)
state = torch.load(student2_path)
model3.load_state_dict(state)

# model4 = models.resnet50(pretrained=False)
# num_ftrs = model4.fc.in_features
# model4.fc = nn.Linear(num_ftrs, 2)
# state = torch.load(student3_path)
# model4.load_state_dict(state)

if __name__ == '__main__':
    model1, model2, model3 = model1.to(device), model2.to(device), model3.to(device)

    model1.eval()
    model2.eval()
    model3.eval()
    model_list = [model1, model2, model3]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embed_model = BertModel.from_pretrained("bert-base-uncased")
    embed_model.eval()

    class_embedding = []
    for cifar_class in classes:
        encoded_input = tokenizer(cifar_class.lower(), return_tensors='pt')
        output = embed_model(**encoded_input)
        class_embedding.append(output[1])
    class_embedding = torch.cat(class_embedding).to(device)

    # Variables to compare teacher and student layers (DC)
    total_pdcgts11_TS1, total_pdcgts21_TS1, total_pdcgts31_TS1, total_pdcgts41_TS1 = 0, 0, 0, 0
    total_pdcgts11_TS2, total_pdcgts21_TS2, total_pdcgts31_TS2, total_pdcgts41_TS2 = 0, 0, 0, 0

    total_pdcgt11, total_pdcgt21, total_pdcgt31, total_pdcgt41 = 0, 0, 0, 0
    total_pdcgt12, total_pdcgt22, total_pdcgt32, total_pdcgt42 = 0, 0, 0, 0
    total_pdcgt13, total_pdcgt23, total_pdcgt33, total_pdcgt43 = 0, 0, 0, 0

    total_pdc11, total_pdc21, total_pdc31 = 0, 0, 0
    total_pdc12, total_pdc22, total_pdc32 = 0, 0, 0
    total_pdc13, total_pdc23, total_pdc33 = 0, 0, 0

    total_pdcPt11, total_pdcPt21, total_pdcPt31, total_pdcPt41 = 0, 0, 0, 0
    total_pdcPt12, total_pdcPt22, total_pdcPt32, total_pdcPt42 = 0, 0, 0, 0

    total_pdcPtF11, total_pdcPtF21, total_pdcPtF31, total_pdcPtF41 = 0, 0, 0, 0
    total_pdcPtF12, total_pdcPtF22, total_pdcPtF32, total_pdcPtF42 = 0, 0, 0, 0

    total_pdcPtSS11, total_pdcPtSS21, total_pdcPtSS31, total_pdcPtSS41 = 0, 0, 0, 0
    total_pdcPtSS12, total_pdcPtSS22, total_pdcPtSS32, total_pdcPtSS42 = 0, 0, 0, 0
    # with torch.no_grad():
    for batch_idx, (inputs, label) in enumerate(testloader):
        images = inputs
        cls = [z.item() for z in label]

        if batch_idx % 10 == 0:
            print(f'At {batch_idx}')

        if len(inputs)>3:
            outputs1, outputs2, outputs3 = [], [], []
            for i, img in enumerate(images):
                # print(img.shape)
                img = np.array(TF.to_pil_image(img))
                img = np.float32(img) / 255
                input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
                targets = [ClassifierOutputSoftmaxTarget(cls[i])]

                for m, model in enumerate(model_list):
                    # target_layers = [model.layer4, model.layer3, model.layer2, model.layer1]
                    target_layers = [model.layer4]
                    with HiResCAM(model=model, target_layers=target_layers) as cam:
                        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

                    cam_metric = ROADLeastRelevantFirst(percentile=50)
                    _, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
                    visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
                    visualization = deprocess_image(visualization)

                    img1 = Image.fromarray(visualization)
                    x = TF.to_tensor(img1)
                    x.unsqueeze_(0)
                    if m == 0:
                        if len(outputs1) == 0:
                            outputs1 = torch.cat((x, x),0)
                        else:
                            outputs1 = torch.cat((outputs1, x), 0)
                    elif m == 1:
                        if len(outputs2) == 0:
                            outputs2 = torch.cat((x, x),0)
                        else:
                            outputs2 = torch.cat((outputs2, x), 0)
                    else:
                        if len(outputs3) == 0:
                            outputs3 = torch.cat((x, x), 0)
                        else:
                            outputs3 = torch.cat((outputs3, x), 0)
            # Remove the first repeated tensor
            outputs1 = outputs1[1:, :, :, :]
            outputs2 = outputs2[1:, :, :, :]
            outputs3 = outputs3[1:, :, :, :]

            # Put to device
            outputs1 = torch.Tensor(outputs1).to(device)
            outputs2 = torch.Tensor(outputs2).to(device)
            outputs3 = torch.Tensor(outputs3).to(device)
            label = label.to(device)

            # Extract Features
            features11 = ResnetFeatureExtractor(2, model1)(outputs1)
            features21 = ResnetFeatureExtractor(3, model1)(outputs1)
            features31 = ResnetFeatureExtractor(4, model1)(outputs1)
            features41 = ResnetFeatureExtractor(5, model1)(outputs1)

            features11 = features11.view([features11.shape[0], -1])
            features21 = features21.view([features21.shape[0], -1])
            features31 = features31.view([features31.shape[0], -1])
            features41 = features41.view([features41.shape[0], -1])

            features12 = ResnetFeatureExtractor(2, model2)(outputs2)
            features22 = ResnetFeatureExtractor(3, model2)(outputs2)
            features32 = ResnetFeatureExtractor(4, model2)(outputs2)
            features42 = ResnetFeatureExtractor(5, model2)(outputs2)

            features12 = features12.view([features12.shape[0], -1])
            features22 = features22.view([features22.shape[0], -1])
            features32 = features32.view([features32.shape[0], -1])
            features42 = features42.view([features42.shape[0], -1])

            features13 = ResnetFeatureExtractor(2, model3)(outputs3)
            features23 = ResnetFeatureExtractor(3, model3)(outputs3)
            features33 = ResnetFeatureExtractor(4, model3)(outputs3)
            features43 = ResnetFeatureExtractor(5, model3)(outputs3)

            features13 = features13.view([features13.shape[0], -1])
            features23 = features23.view([features23.shape[0], -1])
            features33 = features33.view([features33.shape[0], -1])
            features43 = features43.view([features43.shape[0], -1])

            # Get the class embedding from pre-trained BERT model

            ground_truth = class_embedding[label]

            # Comparing information in each layer with the ground truth
            # This is the FSS between the layers of the basemodel and 
            # Student and you can play as you wanted with wich layers to use.
            # ========================================================
            # Comparing Teacher with Student 1
            pdcor1Gt1 = New_DC(features11, features12)
            total_pdcgts11_TS1 += pdcor1Gt1.detach()
            pdcor1Gt2 = New_DC(features21, features22)
            total_pdcgts21_TS1 += pdcor1Gt2.detach()
            pdcor1Gt3 = New_DC(features31, features32)
            total_pdcgts31_TS1 += pdcor1Gt3.detach()
            pdcor1Gt4 = New_DC(features41, features42)
            total_pdcgts41_TS1 += pdcor1Gt4.detach()

            # Comparing Teacher with Student 2
            pdcor1Gt1 = New_DC(features11, features13)
            total_pdcgts11_TS2 += pdcor1Gt1.detach()
            pdcor1Gt2 = New_DC(features21, features23)
            total_pdcgts21_TS2 += pdcor1Gt2.detach()
            pdcor1Gt3 = New_DC(features31, features33)
            total_pdcgts31_TS2 += pdcor1Gt3.detach()
            pdcor1Gt4 = New_DC(features41, features43)
            total_pdcgts41_TS2 += pdcor1Gt4.detach()
            # Teacher
            pdcor1Gt1 = New_DC(features11, ground_truth)
            total_pdcgt11 += pdcor1Gt1.detach()
            pdcor1Gt2 = New_DC(features21, ground_truth)
            total_pdcgt21 += pdcor1Gt2.detach()
            pdcor1Gt3 = New_DC(features31, ground_truth)
            total_pdcgt31 += pdcor1Gt3.detach()
            pdcor1Gt4 = New_DC(features41, ground_truth)
            total_pdcgt41 += pdcor1Gt4.detach()

            # Student1
            pdcor2Gt1 = New_DC(features12, ground_truth)
            total_pdcgt12 += pdcor2Gt1.detach()
            pdcor2Gt2 = New_DC(features22, ground_truth)
            total_pdcgt22 += pdcor2Gt2.detach()
            pdcor2Gt3 = New_DC(features32, ground_truth)
            total_pdcgt32 += pdcor2Gt3.detach()
            pdcor2Gt4 = New_DC(features42, ground_truth)
            total_pdcgt42 += pdcor2Gt4.detach()

            # Student2
            pdcor3Gt1 = New_DC(features13, ground_truth)
            total_pdcgt13 += pdcor3Gt1.detach()
            pdcor3Gt2 = New_DC(features23, ground_truth)
            total_pdcgt23 += pdcor3Gt2.detach()
            pdcor3Gt3 = New_DC(features33, ground_truth)
            total_pdcgt33 += pdcor3Gt3.detach()
            pdcor3Gt4 = New_DC(features43, ground_truth)
            total_pdcgt43 += pdcor3Gt4.detach()

            # Comparing Layers with the last layer
            # ====================================
            # Teacher
            pdcor11 = New_DC(features11, features41)
            total_pdc11 += pdcor11.detach()
            pdcor21 = New_DC(features11, features31)
            total_pdc21 += pdcor21.detach()
            pdcor31 = New_DC(features11, features21)
            total_pdc31 += pdcor31.detach()

            # Student1
            pdcor12 = New_DC(features12, features42)
            total_pdc12 += pdcor12.detach()
            pdcor22 = New_DC(features12, features32)
            total_pdc22 += pdcor22.detach()
            pdcor32 = New_DC(features12, features22)
            total_pdc32 += pdcor32.detach()

            # Student2
            pdcor13 = New_DC(features13, features43)
            total_pdc13 += pdcor13.detach()
            pdcor23 = New_DC(features13, features33)
            total_pdc23 += pdcor23.detach()
            pdcor33 = New_DC(features13, features23)
            total_pdc33 += pdcor33.detach()

            # Partial Distance R(T|S1, GT)
            # ============================
            pdcor1Pt1 = P_DC(features11, features12, ground_truth)
            total_pdcPt11 += pdcor1Pt1.detach()
            pdcor1Pt2 = P_DC(features21, features22, ground_truth)
            total_pdcPt21 += pdcor1Pt2.detach()
            pdcor1Pt3 = P_DC(features31, features32, ground_truth)
            total_pdcPt31 += pdcor1Pt3.detach()
            pdcor1Pt4 = P_DC(features41, features42, ground_truth)
            total_pdcPt41 += pdcor1Pt4.detach()

            # Partial Distance R(S1|T, GT)
            # ============================
            pdcor2Pt1 = P_DC(features12, features11, ground_truth)
            total_pdcPt12 += pdcor2Pt1.detach()
            pdcor2Pt2 = P_DC(features22, features21, ground_truth)
            total_pdcPt22 += pdcor2Pt2.detach()
            pdcor2Pt3 = P_DC(features32, features31, ground_truth)
            total_pdcPt32 += pdcor2Pt3.detach()
            pdcor2Pt4 = P_DC(features42, features41, ground_truth)
            total_pdcPt42 += pdcor2Pt4.detach()

            # Partial Distance R(T|S2, GT)
            # ============================
            pdcor1PtF1 = P_DC(features11, features13, ground_truth)
            total_pdcPtF11 += pdcor1PtF1.detach()
            pdcor1PtF2 = P_DC(features21, features23, ground_truth)
            total_pdcPtF21 += pdcor1PtF2.detach()
            pdcor1PtF3 = P_DC(features31, features33, ground_truth)
            total_pdcPtF31 += pdcor1PtF3.detach()
            pdcor1PtF4 = P_DC(features41, features43, ground_truth)
            total_pdcPtF41 += pdcor1PtF4.detach()

            # Partial Distance R(S2|T, GT)
            # ============================
            pdcor2PtF1 = P_DC(features13, features11, ground_truth)
            total_pdcPtF12 += pdcor2PtF1.detach()
            pdcor2PtF2 = P_DC(features23, features21, ground_truth)
            total_pdcPtF22 += pdcor2PtF2.detach()
            pdcor2PtF3 = P_DC(features33, features31, ground_truth)
            total_pdcPtF32 += pdcor2PtF3.detach()
            pdcor2PtF4 = P_DC(features43, features41, ground_truth)
            total_pdcPtF42 += pdcor2PtF4.detach()

            # Partial Distance R(S1|S2, GT)
            # ============================
            pdcor1PtF1 = P_DC(features12, features13, ground_truth)
            total_pdcPtSS11 += pdcor1PtF1.detach()
            pdcor1PtF2 = P_DC(features22, features23, ground_truth)
            total_pdcPtSS21 += pdcor1PtF2.detach()
            pdcor1PtF3 = P_DC(features32, features33, ground_truth)
            total_pdcPtSS31 += pdcor1PtF3.detach()
            pdcor1PtF4 = P_DC(features42, features43, ground_truth)
            total_pdcPtSS41 += pdcor1PtF4.detach()

            # Partial Distance R(S2|S1, GT)
            # ============================
            pdcor2PtF1 = P_DC(features13, features12, ground_truth)
            total_pdcPtSS12 += pdcor2PtF1.detach()
            pdcor2PtF2 = P_DC(features23, features22, ground_truth)
            total_pdcPtSS22 += pdcor2PtF2.detach()
            pdcor2PtF3 = P_DC(features33, features32, ground_truth)
            total_pdcPtSS32 += pdcor2PtF3.detach()
            pdcor2PtF4 = P_DC(features43, features42, ground_truth)
            total_pdcPtSS42 += pdcor2PtF4.detach()

        break
    from contextlib import redirect_stdout

    with open('distance/comparingModelsPetImagesNewNoPre/CAM1.txt', 'w') as f:
        with redirect_stdout(f):

            #Check the FSS



            print(f'Similarity between Teacher and FeatureKD layer[Reverse order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcgts11_TS1 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcgts21_TS1 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcgts31_TS1 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcgts41_TS1 / len(testloader)
            print(model1_info_rem3)

            print(f'Similarity between Teacher and AttentionKD layer[Reverse order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcgts11_TS2 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcgts21_TS2 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcgts31_TS2 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcgts41_TS2 / len(testloader)
            print(model1_info_rem3)

            print(f'Comparing Teacher layers with last layer[same order]')
            print(f'====================================')
            model1_info_rem1 = total_pdc11 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdc21 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdc31 / len(testloader)
            print(model1_info_rem3)

            print(f'Comparing FeatureKD layers with last layer[same order]')
            print(f'====================================')
            model2_info_rem1 = total_pdc12 / len(testloader)
            print(model2_info_rem1)
            model2_info_rem2 = total_pdc22 / len(testloader)
            print(model2_info_rem2)  #
            model2_info_rem3 = total_pdc32 / len(testloader)
            print(model2_info_rem3)

            print(f'Comparing AttentionKd layers with last layer[same order]')
            print(f'====================================')
            model2_info_rem1 = total_pdc13 / len(testloader)
            print(model2_info_rem1)
            model2_info_rem2 = total_pdc23 / len(testloader)
            print(model2_info_rem2)  #
            model2_info_rem3 = total_pdc33 / len(testloader)
            print(model2_info_rem3)

            print(f'Comparing Teacher layers with ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcgt11 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcgt21 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcgt31 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcgt41 / len(testloader)
            print(model1_info_rem3)

            print(f'Comparing FeatureKD layers with ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcgt12 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcgt22 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcgt32 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcgt42 / len(testloader)
            print(model1_info_rem3)

            print(f'Comparing AttentionKD layers with ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcgt13 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcgt23 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcgt33 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcgt43 / len(testloader)
            print(model1_info_rem3)

            print(f'Teacher conditioned on FeatureKD and ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcPt11 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcPt21 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcPt31 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcPt41 / len(testloader)
            print(model1_info_rem3)

            print(f'Teacher Conditioned on AttentionKD and ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcPtF11 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcPtF21 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcPtF31 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcPtF41 / len(testloader)
            print(model1_info_rem3)

            print(f'FeatureKD Conditioned on Teacher and ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcPt12 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcPt22 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcPt32 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcPt42 / len(testloader)
            print(model1_info_rem3)

            print(f'AttentionKD Conditioned on Teacher and ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcPtF12 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcPtF22 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcPtF32 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcPtF42 / len(testloader)
            print(model1_info_rem3)

            print(f'FeatureKD Conditioned on AttentionKd  and ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcPtSS11 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcPtSS21 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcPtSS31 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcPtSS41 / len(testloader)
            print(model1_info_rem3)

            print(f'AttentionKD Conditioned on FeatureKD and ground_truth[Reverse Order]')
            print(f'====================================')
            model1_info_rem1 = total_pdcPtSS12 / len(testloader)
            print(model1_info_rem1)
            model1_info_rem2 = total_pdcPtSS22 / len(testloader)
            print(model1_info_rem2)  #
            model1_info_rem3 = total_pdcPtSS32 / len(testloader)
            print(model1_info_rem3)
            model1_info_rem3 = total_pdcPtSS42 / len(testloader)
            print(model1_info_rem3)