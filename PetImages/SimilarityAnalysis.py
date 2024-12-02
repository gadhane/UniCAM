import torch

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random
import os

from torch_cka import CKA
import torch
import os

import torchvision
import torch.nn as nn
import models.MobileNet as Mov
import models.ResNet as ResNet

import warnings
warnings.filterwarnings("ignore")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

data_dir = '/data/ILSVRC12/'
batch_size = 120

valdir = os.path.join(data_dir, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(valdir,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize])
                                   )
testloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model1 = ResNet.resnet152(pretrained=True)

Student_path = 'ImageNet__Vanila_KD_160Epoches.pt'
model2 = ResNet.resnet50(pretrained=False)
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(Student_path)
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model2.load_state_dict(new_state_dict)

if __name__ == '__main__':
    print(f'inside main')
    # Comparing Teacher and Feature based student
    cka = CKA(model1, model1,
            model1_name="Teacher", model2_name="Teacher",
            device='cuda')
    cka.compare(testloader)
    cka.plot_results(save_path="Teacher.png")
    results = cka.export()
    dt = results['CKA'].numpy()
    df = pd.DataFrame(dt)
    df.to_csv("SimTeacher_Teacher.csv",header=False, index=False)

    # Comparing Teacher and Response based student
    cka = CKA(model2, model2,
            model1_name="Response Std", model2_name="Response Std",
            device='cuda')
    cka.compare(testloader)
    cka.plot_results(save_path="Response based KD.png")
    results = cka.export()
    dt = results['CKA'].numpy()
    df = pd.DataFrame(dt)
    df.to_csv("SimStudent_Student.csv", header=False, index=False)
    # Comparing Feature and Response based students
    cka = CKA(model1, model2,
            model1_name="Teacher", model2_name="Student",
            device='cuda')
    cka.compare(testloader)
    cka.plot_results(save_path="Teacher vs Student.png")
    results = cka.export()
    dt = results['CKA'].numpy()
    df = pd.DataFrame(dt)
    df.to_csv("SimTeacher_Student.csv", header=False, index=False)

