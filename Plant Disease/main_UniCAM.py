from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import os
import torch
import numpy as np
import cv2
from torch.nn import functional as F
from torch import nn

from AdvGradCAM import AdGradCAM
from GradCAM import GradCAM
from pdCor import *
from utils import *

import warnings

warnings.filterwarnings("ignore")


# rest of the code

def normalize(input):
    output = (input - np.min(input)) / (np.max(input) - np.min(input))
    return output


def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))

    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    # cv2.imshow(gcam)

    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))
    cv2.rectangle(gcam, (384, 0), (510, 128), (0, 255, 0), 3)
    # plt.imshow(np.uint8(gcam))
    # plt.show()

def get_normalize_layer(_IMAGENET_MEAN, _IMAGENET_STDDEV):
    return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)

def save_raw_image(filename, raw_image):
    cv2.imwrite(filename, np.uint8(raw_image))


# input = 'CAMS/17.jpg'
gcamoutput = 'gradcams'  # Directory for the visualization result
adgcamoutput = 'adgradcams'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
           'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
           'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
           'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy',
           'Cherry_(including_sour)___Powdery_mildew',
           'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy',
           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus',
           'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight',
           'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)',
           'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite',
           'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']

if os.path.isdir(gcamoutput):
    print('Output directory exists')
else:
    os.mkdir(gcamoutput)

if os.path.isdir(adgcamoutput):
    print('Output directory exists')
else:
    os.mkdir(adgcamoutput)

weights_T = 'models/PlantDisease/teacher.pt'
weights_S = 'models/PlantDisease/student.pt'
#
model_T = models.resnet50(pretrained=False)
num_ftrs = model_T.fc.in_features
model_T.fc = nn.Linear(num_ftrs, 38)
state = torch.load(weights_T)
model_T.load_state_dict(state)
model_T.eval()
model_T.to(device)

model_S = models.resnet50(pretrained=False)
num_ftrs = model_S.fc.in_features
model_S.fc = nn.Linear(num_ftrs, 38)
state = torch.load(weights_S)
model_S.load_state_dict(state)
model_S.eval()
model_S.to(device)

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

### Target layers (VGG16)
# targets = ['layer3.5.conv3', 'layer4.2.conv3']
cams_folder = 'CAMS'  # Directory containing the images

normalize_X = get_normalize_layer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
normalize_Y = get_normalize_layer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
block = 1  # Layers is always reverse, Eg. 2 is the layer4, 5 is layer1.
# modelTS1remain = PDC_Model(model_T, model_S, normalize_X, normalize_Y, block)
modelTS1remain = PDC_Model(model_S, model_T, normalize_X, normalize_Y, block)

targets = ['modelX.layer3.5.conv3', 'modelX.layer4.2.conv3']

if __name__ == '__main__':
    # Loop through each file in the CAMS directory
    for image_file in os.listdir(cams_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_image = os.path.join(cams_folder, image_file)

            print('=' * 20)
            print(f"Processing image \'{input_image}\'")

            raw_image = cv2.imread(input_image)[..., ::-1]
            im_pil = Image.fromarray(raw_image)

            image = data_transforms(im_pil).unsqueeze(0)
            a = transforms.ToPILImage()(image.squeeze())

            tr_image = np.asarray(np.transpose(image.squeeze(0), (1, 2, 0)))
            tr_image = cv2.cvtColor(tr_image, cv2.COLOR_BGR2RGB)
            tr_image = np.uint8(normalize(tr_image) * 255)
            # image_name = input_image.split(sep="/")[-1].split(sep=".")[0]
            image_name = input_image.split(sep=os.path.sep)[-1].split(sep=".")[0]

            save_dir = os.path.join(gcamoutput, image_name)
            adsave_dir = os.path.join(adgcamoutput, image_name)

            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            if not os.path.isdir(adsave_dir):
                os.mkdir(adsave_dir)

            a.save(os.path.join(save_dir, '_input_image.jpg'))
            a.save(os.path.join(adsave_dir, '_input_image.jpg'))

            adgcam = AdGradCAM(model=modelTS1remain)
            gcam = GradCAM(model=modelTS1remain)
            print('-------------------')

            probs, idx = gcam.forward(image.to(device))
            # print(f'adgcam{probs}')
            # print(f'gcam:{idx}')

            adprobs, adidx = adgcam.forward(image.to(device))

            for j in range(0, len(targets)):
                gcam.backward(idx=idx[0])
                output = gcam.generate(target_layer=targets[j])
                # Filename : {ClassName}_gcam_{NumLayer}
                print(f'-----------------------')
                print(classes[idx[0].item()])
                print(f'-----------------------')
                save_gradcam(save_dir + '/{}_gcam_{}.png'.format(classes[idx[0].item()], targets[j]), output, tr_image)

            for i in range(0, len(targets)):
                adgcam.backward(idx=adidx[0])
                output = adgcam.generate(target_layer=targets[i])
                # Filename : {ClassName}_gcam_{NumLayer}
                print(f'-----------------------')
                print(classes[idx[0].item()])
                print(f'-----------------------')
                save_gradcam(adsave_dir + '/{}_gcam_{}.png'.format(classes[adidx[0].item()], targets[i]), output, tr_image)
            break
            print(f'______________________________________________________________')
