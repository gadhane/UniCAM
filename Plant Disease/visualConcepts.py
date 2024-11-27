import argparse

import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pdb
import cv2


from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenGradCAM, HiResCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst

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
from pdCor_grad import *

import shutup
shutup.please()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PLANT_IMG_LOC = 'Images'

basemodel = 'models/PlantDisease/basemodel.pt'
student1_path = 'models/PlantDisease/student.pt'


model1 = models.resnet50(pretrained=False)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 38)
state = torch.load(basemodel)
model1.load_state_dict(state)

model2 = models.resnet50(pretrained=False)
num_ftrs = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs, 38)
state = torch.load(student1_path)
model2.load_state_dict(state)

class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, i, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-i])

    def __call__(self, x):
        return self.feature_extractor(x)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='eigengradcam',
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
    return 2


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
    elif split == "cam":
        subdir = os.path.join(dir, "cam")
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

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
     'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
     'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
     'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
     'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
     'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy',
     'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
     'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
     'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
     'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

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
    test_dataset = get_dataset('cam')

    pin_memory = True
    torch.manual_seed(4)
    test_loader = DataLoaderX(test_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=pin_memory)

    num_classes = 38
    block = 1#Layers is always reverse, Eg. 2 is the layer4, 5 is layer1.

    modelTS1remain = PDC_Model(modelT, modelS1, normalize_X, normalize_Y, block)
    modelS1remain = PDC_Model(modelS1, modelT, normalize_X, normalize_Y, block)

    target_layers_TS1 = [modelTS1remain.modelX.layer4[-1]]
    target_layers_S1 = [modelS1remain.modelX.layer4[-1]]

    target_T = [model1.layer4[-1]]
    target_S1 = [model2.layer4[-1]]

    ImageNetLabelEmbedding = torch.load('ImageNet_Class_Embedding.pt')
    ImageNetLabelEmbedding = ImageNetLabelEmbedding.to(device)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        input_tensor = inputs.cuda()
        cls = [z.item() for z in targets]
        # R^2(T|S1, GT)
        cam_TS1 = PDC_CAM(model=modelTS1remain, target_layers=target_layers_TS1, use_cuda=True,
                          reshape_transform=None)
        grayscale_cam_TS1 = cam_TS1(input_tensor=input_tensor, targets=None)
        print(f'Checking TS1 Done..')
        # R^2(S1|T, GT)
        cam_S1 = PDC_CAM(model=modelS1remain, target_layers=target_layers_S1, use_cuda=True,
                         reshape_transform=None)
        grayscale_cam_S1 = cam_S1(input_tensor=input_tensor, targets=None)
        print(f'Checking S1 Done..')
        cdir = 'Distilled Features/VisualConcepts/'
        if not os.path.exists(cdir):
            os.mkdir(cdir)
        methods = {"gradcam": HiResCAM}
        cam_algorithm = methods[args.method]
        for idx in range(len(targets)):
            rgb_img = inputs[idx].numpy().transpose(1, 2, 0)
            imgcp = rgb_img.copy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_img = std * rgb_img + mean
            rgb_img = np.clip(rgb_img, 0, 1)
            rgb_img = np.float32(rgb_img)
            input_T = preprocess_image(rgb_img)
            tar = None
            # GradCAM Model1
            targets = [ClassifierOutputSoftmaxTarget(cls[idx])]

            rgb_img = inputs.permute(0, 2, 3, 1).numpy()
            rgb_img = rgb_img[idx, :]
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'.jpg', rgb_img * 255)

            # ===================================
            cam_metric = ROADLeastRelevantFirst(percentile=50)
            htmap = np.asarray(grayscale_cam_TS1[idx:idx+1, ...])
            _, visualizations = cam_metric(input_T.to(device), htmap, targets, model1,
                                           return_visualization=True)
            visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
            visualization = deprocess_image(visualization)
            cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'_TS1_visconc.jpg', visualization)
            grayscale_cam_idx = grayscale_cam_TS1[idx, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam_idx, use_rgb=False)
            cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'_TS1_cam.jpg', visualization)

            # ===================================
            cam_metric = ROADLeastRelevantFirst(percentile=50)
            htmap = np.asarray(grayscale_cam_S1[idx:idx + 1, ...])
            _, visualizations = cam_metric(input_T.to(device), htmap, targets, model2,
                                           return_visualization=True)
            visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
            visualization = deprocess_image(visualization)
            cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'_S1T_visconc.jpg', visualization)

            grayscale_cam_X_idx = grayscale_cam_S1[idx, :]
            visualization_X = show_cam_on_image(rgb_img, grayscale_cam_X_idx, use_rgb=False)
            cv2.imwrite(str(cdir) + str(batch_idx) + str(idx) + f'_S1T_cam.jpg', visualization_X)
            # ===================================
        if batch_idx == 10:
            break