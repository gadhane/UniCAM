# '''
# Adapted from Original Author: zhenxingjian
# https://github.com/zhenxingjian/Partial_Distance_Correlation/blob/main/Partial_Distance_Correlation/main_CAM.py
# '''

import argparse
import numpy as np
import time
import os
import pdb
import cv2


from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenGradCAM, HiResCAM, \
    EigenCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

from pdCor import *
from utils import *

from pdCor_model import *
from pdCor_CAM import *

import shutup

shutup.please()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PLANT_IMG_LOC = 'Images'

basemodel_path = 'models/PlantDisease/basemodel.pt'
student1_path = 'models/PlantDisease/student.pt'


model1 = models.resnet50(pretrained=False)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 38)
state = torch.load(basemodel_path)
model1.load_state_dict(state)

model2 = models.resnet50(pretrained=False)
num_ftrs = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs, 38)
state = torch.load(student1_path)
model2.load_state_dict(state)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()

    return args

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_dataset(split):
    """Return the dataset as a PyTorch Dataset object"""
    return _imagenet(split)


def get_num_classes():
    """Return the number of classes in the dataset. """
    return 38


def get_normalize_layer(_IMAGENET_MEAN, _IMAGENET_STDDEV):
    return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)


def _imagenet(split):
    dir = PLANT_IMG_LOC
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])
    elif split == "cam2":
        subdir = os.path.join(dir, "cam2")
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])
    return datasets.ImageFolder(subdir, transform)

def reshape_transform(tensor, height=7, width=7):
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
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')

    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()

    modelT = model1
    modelS1 = model2

    normalize_X = get_normalize_layer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalize_Y = get_normalize_layer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    test_dataset = get_dataset('cam2')

    pin_memory = True
    torch.manual_seed(4)
    test_loader = DataLoaderX(test_dataset, shuffle=False, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=pin_memory)

    num_classes = 38
    block = 1#Layers is always reverse, Eg. 2 is the layer4, 5 is layer1.

    modelTS1remain = PDC_Model(modelT, modelS1, normalize_X, normalize_Y, block)
    modelS1remain = PDC_Model(modelS1, modelT, normalize_X, normalize_Y, block)

    target_layers_TS1 = [modelTS1remain.modelX.layer1[-3]]
    target_layers_S1 = [modelS1remain.modelX.layer1[-3]]

    target_T = [model1.layer3[-1]]
    target_S1 = [model2.layer3[-1]]

    ImageNetLabelEmbedding = torch.load('ImageNet_Class_Embedding.pt')
    ImageNetLabelEmbedding = ImageNetLabelEmbedding.to(device)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        input_tensor = inputs.cuda()
        #convert targets to imagenet indexing
        #Modify and uncomment the following line of code based on your target values.
        # t1 = [281 if i == 0 else 253 for i in targets]
        t1 = torch.tensor(t1)
        target_category = ImageNetLabelEmbedding[t1]

        # R^2(T|S1, GT)
        cam_TS1 = PDC_CAM(model=modelTS1remain, target_layers=target_layers_TS1, use_cuda=True,
                          reshape_transform=None)
        grayscale_cam_TS1 = cam_TS1(input_tensor=input_tensor, targets=None)

        print(f'Checking T|S1 Done..')
        
        # R^2(S1|T, GT)
        cam_S1 = PDC_CAM(model=modelS1remain, target_layers=target_layers_S1, use_cuda=True,
                         reshape_transform=None)
        grayscale_cam_S1 = cam_S1(input_tensor=input_tensor, targets=None)
        print(f'Checking S1|T Done..')
        # R^2(S2|T, GT)
        cdir = 'Distilled Features/Potato___Early_blight1/'
        if not os.path.exists(cdir):
            os.mkdir(cdir)
        methods = {"gradcam": GradCAM}

        cam_algorithm = methods[args.method]
        for idx in range(len(targets)):
            rgb_img = inputs[idx].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_img = std * rgb_img + mean
            rgb_img = np.clip(rgb_img, 0, 1)
            rgb_img = np.float32(rgb_img)
            input_T = preprocess_image(rgb_img)
            # tar = [ClassifierOutputSoftmaxTarget(5)]
            tar = None
            # # GradCAM Model1
            with cam_algorithm(model=model1,
                               target_layers=target_T,
                               use_cuda=True) as cam:
                # cam.batch_size = 32
                grayscale_cam = cam(input_tensor=input_T,
                                    targets=tar,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)
                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'_OrigT_cam.jpg', cam_image)
            
            # GradCAM Model2
            with cam_algorithm(model=model2,
                               target_layers=target_S1,
                               use_cuda=True) as cam:
                # cam.batch_size = 32
                grayscale_cam = cam(input_tensor=input_T,
                                    targets=tar,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)
                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'_OrigS1_cam.jpg', cam_image)
            
            rgb_img = inputs.permute(0, 2, 3, 1).numpy()
            rgb_img = rgb_img[idx, :]
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'.jpg', rgb_img * 255)

            grayscale_cam_idx = grayscale_cam_TS1[idx, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam_idx, use_rgb=False)
            cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'_TS1_cam.jpg', visualization)


            grayscale_cam_X_idx = grayscale_cam_S1[idx, :]
            visualization_X = show_cam_on_image(rgb_img, grayscale_cam_X_idx, use_rgb=False)
            cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'_S1T_cam.jpg', visualization_X)


        # if batch_idx == 10:
        #     break