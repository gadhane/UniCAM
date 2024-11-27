# '''
# Adapted from Original Author zhenxingjian
# https://github.com/zhenxingjian/Partial_Distance_Correlation/blob/main/Partial_Distance_Correlation/PDC_model.py
# '''
import argparse
import numpy as np
import time
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

def P_Distance_Matrix(latent):
    n = latent.shape[0]
    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim=-1) + 1e-18)
    matrix_A = matrix_a - torch.sum(matrix_a, dim=0, keepdims=True) / (n - 2) - torch.sum(matrix_a, dim=1,
                                                                                          keepdims=True) / (n - 2) \
               + torch.sum(matrix_a) / ((n - 1) * (n - 2))

    diag_A = torch.diag(torch.diag(matrix_A))
    matrix_A = matrix_A - diag_A
    return matrix_A


def Distance_Correlation(latent, control):
    latent = F.normalize(latent)
    control = F.normalize(control)

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
    # correlation_r = torch.pow(Gamma_XY,2)/(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r


def P_Distance_Matrix(latent):
    n = latent.shape[0]
    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim=-1) + 1e-18)
    matrix_A = matrix_a - torch.sum(matrix_a, dim=0, keepdims=True) / (n - 2) - torch.sum(matrix_a, dim=1,
                                                                                          keepdims=True) / (n - 2) \
               + torch.sum(matrix_a) / ((n - 1) * (n - 2))

    diag_A = torch.diag(torch.diag(matrix_A))
    matrix_A = matrix_A - diag_A
    return matrix_A


def bracket_op(matrix_A, matrix_B):
    n = matrix_A.shape[0]
    return torch.sum(matrix_A * matrix_B) / (n * (n - 3))


def P_removal(matrix_A, matrix_C):
    result = matrix_A - bracket_op(matrix_A, matrix_C) / bracket_op(matrix_C, matrix_C) * matrix_C
    return result


def Correlation(matrix_A, matrix_B):
    Gamma_XY = bracket_op(matrix_A, matrix_B)
    Gamma_XX = bracket_op(matrix_A, matrix_A)
    Gamma_YY = bracket_op(matrix_B, matrix_B)

    correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-18)

    return correlation_r


def P_DC(latent_A, latent_B, ground_truth):
    matrix_A = P_Distance_Matrix(latent_A)
    matrix_B = P_Distance_Matrix(latent_B)
    matrix_GT = P_Distance_Matrix(ground_truth)

    # breakpoint()

    matrix_A_B = P_removal(matrix_A, matrix_B)
    # breakpoint()
    cr = Correlation(matrix_A_B, matrix_GT)

    return cr


def similarity(latent_A, ground_truth):
    matrix_A = P_Distance_Matrix(latent_A)
    matrix_GT = P_Distance_Matrix(ground_truth)
    cr = Correlation(matrix_A, matrix_GT)

    return cr


class Loss_DC(nn.Module):
    def __init__(self, alpha=0.1):
        super(Loss_DC, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.ImageNetLabelEmbedding = torch.nn.parameter.Parameter(torch.load('ImageNet_Class_Embedding.pt'),
                                                                   requires_grad=False)
        print("Loss balance alpha is: ", alpha)

    def CE(self, logit, target):
        return self.ce(logit, target)

    def forward(self, outputs, featuresX, featuresY, targets):
        cls_loss = self.CE(outputs, targets)

        imagenet_embedding = self.ImageNetLabelEmbedding[targets]
        dc_loss = P_DC(featuresX, featuresY, imagenet_embedding)

        loss = cls_loss - self.alpha * dc_loss
        return loss, cls_loss, dc_loss

class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, i, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-i])

    def __call__(self, x):
        return self.feature_extractor(x)

class PDC_Model(nn.Module):
    def __init__(self, modelX, modelY, normalize_X, normalize_Y, layer):
        super(PDC_Model, self).__init__()
        self.modelX = modelX
        self.modelY = modelY
        self.normalize_X = normalize_X
        self.normalize_Y = normalize_Y
        self.layer = layer

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputsX = self.normalize_X(inputs)
        # featuresX = self.modelX.forward_features(inputsX)
        featuresX = ResnetFeatureExtractor(self.layer, self.modelX)(inputsX)
        # featuresX =  self.modelX(inputsX)
        featuresX = featuresX.reshape([batch_size, -1])

        inputsY = self.normalize_Y(inputs)
        # featuresY = self.modelY.forward_features(inputsY)
        # featuresY = self.modelY.global_pool(featuresY)
        featuresY = ResnetFeatureExtractor(self.layer, self.modelY)(inputsY)
        featuresY = featuresY.reshape([batch_size, -1])

        matrix_A = P_Distance_Matrix(featuresX)
        matrix_B = P_Distance_Matrix(featuresY)


        matrix_A_B = P_removal(matrix_A, matrix_B)

        return matrix_A_B


class Model2Matrix(nn.Module):
    def __init__(self, modelX, normalize_X, layer):
        super(Model2Matrix, self).__init__()
        self.modelX = modelX
        self.normalize_X = normalize_X
        self.block = layer
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputsX = self.normalize_X(inputs)
        # featuresX = self.modelX.forward_features(inputsX)
        featuresX = ResnetFeatureExtractor(self.block, self.modelX)(inputsX)
        featuresX = featuresX.reshape([batch_size, -1])

        matrix_A = P_Distance_Matrix(featuresX)

        return matrix_A