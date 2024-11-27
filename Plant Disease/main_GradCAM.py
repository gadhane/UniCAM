
# This code generates Grad-CAM visualizations for a given image and model.
# Based on the Target class.

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
    #cv2.imshow(gcam)
    
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))
    cv2.rectangle(gcam,(384,0),(510,128),(0,255,0),3)
    # plt.imshow(np.uint8(gcam))
    # plt.show()

def save_raw_image(filename, raw_image):
    cv2.imwrite(filename, np.uint8(raw_image))


input = 'CAMS/09.jpg'
output = 'gradcams'#Directory for the visualization result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
        'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 
        'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
        'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
        'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
        'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 
        'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 
        'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']

if os.path.isdir(output):
    print('Output directory exists')
else:
    os.mkdir(output)



weights = 'models/PlantDisease/basemodel.pt'
# weights = 'models/PlantDisease/student.pt'

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 38)
state = torch.load(weights)
model.load_state_dict(state)
model.eval()
model.to(device)


data_transforms = transforms.Compose([
    transforms.ToTensor()
])

### Target layers (VGG16)
targets = ['layer3.5.conv3', 'layer4.2.conv3']
image_num = 0
input_image = input

print('='*20)
print(f"Processing image \'{input_image}\'")

raw_image = cv2.imread(input_image)[..., ::-1]
im_pil = Image.fromarray(raw_image)

image = data_transforms(im_pil).unsqueeze(0)
a = transforms.ToPILImage()(image.squeeze())

tr_image = np.asarray(np.transpose(image.squeeze(0),(1,2,0)))
tr_image = cv2.cvtColor(tr_image,cv2.COLOR_BGR2RGB)
tr_image = np.uint8(normalize(tr_image)*255)
image_name = input_image.split(sep="/")[-1].split(sep=".")[-2]

save_dir = os.path.join(output,image_name)

if os.path.isdir(save_dir):
    print('Save directory exists')
else:
    os.mkdir(save_dir)


a.save(os.path.join(save_dir,'_input_image'+'.jpg'))

gcam = GradCAM(model = model)

probs, idx = gcam.forward(image.to(device))
for j in range(0,len(targets)):
    gcam.backward(idx=idx[0])
    output = gcam.generate(target_layer=targets[j])
    # Filename : {ClassName}_gcam_{NumLayer}
    print(classes[idx[0].item()])
    save_gradcam(save_dir+'/{}_gcam_{}.png'.format(classes[idx[0].item()], targets[j]), output, tr_image)
