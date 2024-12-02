from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import numpy as np
import time
import os
import pdb
import cv2

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


from pdCor import *
from utils import *

from pdCor_model import *
from pdCor_CAM import *

import shutup

shutup.please()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGENET_LOC_ENV = 'D:/ImageNet/PetImages/PetImages'

# model1_path = 'models/student_Vanilla_ResNet50_ResNet101.pt'
# model2_path = 'models/student_attKd_ResNet50_ResNet50.pt'
# model3_path = 'models/student_vanilla_ResNet50_Vgg19.pt'

teacher_path = 'models/PetImages_ModelResNet50.pt'
student1_path = 'models/student_vanila_ResNet50_ResNet50.pt'
student2_path = 'models/student_Feature_ResNet50_ResNet50.pt'
student3_path = 'models/student_attKd_ResNet50_ResNet50.pt'


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

model4 = models.resnet50(pretrained=False)
num_ftrs = model4.fc.in_features
model4.fc = nn.Linear(num_ftrs, 2)
state = torch.load(student3_path)
model4.load_state_dict(state)

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_dataset(split):
    """Return the dataset as a PyTorch Dataset object"""
    return _imagenet(split)


def get_num_classes():
    """Return the number of classes in the dataset. """
    return 2


def get_normalize_layer(_IMAGENET_MEAN, _IMAGENET_STDDEV):
    return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)


def _imagenet(split):
    dir = IMAGENET_LOC_ENV
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])
    elif split == "test":
        subdir = os.path.join(dir, "test")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])
    return datasets.ImageFolder(subdir, transform)

def reshape_transform(tensor, height=7, width=7):
    print(tensor.shape)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Partial Distance Correlation Grad-CAM')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')

    args = parser.parse_args()

    modelT = model1
    modelS1 = model2
    modelS2 = model3
    modelS3 = model4

    normalize_X = get_normalize_layer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalize_Y = get_normalize_layer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    test_dataset = get_dataset('test')

    pin_memory = True
    torch.manual_seed(5)
    test_loader = DataLoaderX(test_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=pin_memory)

    num_classes = 2
    modelTS1remain = PDC_Model(modelT, modelS1, normalize_X, normalize_Y)
    modelTS2remain = PDC_Model(modelT, modelS2, normalize_X, normalize_Y)
    modelTS3remain = PDC_Model(modelT, modelS3, normalize_X, normalize_Y)
    model_X_DC = Model2Matrix(old_modelX, normalize_X)
    # model_Y_DC = Model2Matrix(modelY, normalize_Y)
    modelS1remain = PDC_Model(modelS1, modelT, normalize_X, normalize_Y)
    modelS2remain = PDC_Model(modelS2, modelT, normalize_X, normalize_Y)
    modelS3remain = PDC_Model(modelS3, modelT, normalize_X, normalize_Y)

    target_layers_TS1 = [modelTS1remain.modelX.layer3[-1].conv3]
    target_layers_TS2 = [modelTS2remain.modelX.layer3[-1].conv3]
    target_layers_TS3 = [modelTS3remain.modelX.layer3[-1].conv3]
    target_layers_S1 = [modelS1remain.modelX.layer3[-1].conv3]
    target_layers_S2 = [modelS2remain.modelX.layer3[-1].conv3]
    target_layers_S3 = [modelS3remain.modelX.layer3[-1].conv3]

    ImageNetLabelEmbedding = torch.load('ImageNet_Class_Embedding.pt')
    ImageNetLabelEmbedding = ImageNetLabelEmbedding.to(device)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        input_tensor = inputs.cuda()
        #convert targets to imagenet indexing
        t1 = [281 if i == 0 else 253 for i in targets]
        t1 = torch.tensor(t1)
        target_category = ImageNetLabelEmbedding[t1]

        # R^2(T|S1, GT)
        cam_TS1 = PDC_CAM(model=modelTS1remain, target_layers=target_layers_TS1, use_cuda=True,
                      reshape_transform=None)
        grayscale_cam_TS1 = cam_TS1(input_tensor=input_tensor, targets=target_category)

        # R^2(T|S2, GT)
        cam_TS2 = PDC_CAM(model=modelTS2remain, target_layers=target_layers_TS2, use_cuda=True,
                          reshape_transform=None)
        grayscale_cam_TS2 = cam_TS2(input_tensor=input_tensor, targets=target_category)

        # R^2(T|S3, GT)
        cam_TS3 = PDC_CAM(model=modelTS3remain, target_layers=target_layers_TS3, use_cuda=True,
                          reshape_transform=None)
        grayscale_cam_TS3 = cam_TS3(input_tensor=input_tensor, targets=target_category)

        # R^2(S1|T, GT)
        cam_S1 = PDC_CAM(model=modelS1remain, target_layers=target_layers_S1, use_cuda=True,
                        reshape_transform=None)
        grayscale_cam_S1 = cam_S1(input_tensor=input_tensor, targets=target_category)

        # R^2(S2|T, GT)
        cam_S2 = PDC_CAM(model=modelS2remain, target_layers=target_layers_S2, use_cuda=True)
        grayscale_cam_S2 = cam_S2(input_tensor=input_tensor, targets=target_category)

        # R^2(S3|T, GT)
        cam_S3 = PDC_CAM(model=modelS3remain, target_layers=target_layers_S3, use_cuda=True)
        grayscale_cam_S3 = cam_S3(input_tensor=input_tensor, targets=target_category)

        if not os.path.exists('RemoveResults/Layer1/'):
            os.mkdir('RemoveResults/Layer1/')
        for idx in range(len(targets)):
            rgb_img = inputs.permute(0, 2, 3, 1).numpy()
            rgb_img = rgb_img[idx, :]
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            cv2.imwrite('RemoveResults/Layer1/' + str(batch_idx) + str(idx) + f'.jpg', rgb_img * 255)
            grayscale_cam_idx = grayscale_cam_T[idx, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam_idx, use_rgb=False)
            cv2.imwrite('RemoveResults/Layer1/' + str(batch_idx) + str(idx) + f'_TS1_cam.jpg', visualization)

            grayscale_cam_X_idx = grayscale_cam_S1[idx, :]
            visualization_X = show_cam_on_image(rgb_img, grayscale_cam_X_idx, use_rgb=False)
            cv2.imwrite('RemoveResults/Layer1/' + str(batch_idx) + str(idx) + f'_S1T_cam.jpg', visualization_X)

            grayscale_cam_Y_idx = grayscale_cam_S2[idx, :]
            visualization_Y = show_cam_on_image(rgb_img, grayscale_cam_Y_idx, use_rgb=False)

            cv2.imwrite('RemoveResults/Layer1/' + str(batch_idx) + str(idx) + f'_S2T_cam.jpg', visualization_Y)

        if batch_idx == 10:
            break