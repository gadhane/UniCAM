import argparse
import numpy as np
import time
import os
import pdb

from pathlib import Path
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch
from torchmetrics import Metric, BootStrapper
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 30})
if __name__ == '__main__':
    PATH = 'distance/comparingModelsPetImagesNewNoPre/Case3/'

    sim_mat = np.load(PATH + 'modelX_model_X.npy')

    figure(figsize=(6, 6), dpi=60)
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('Teacher', fontsize=50)
    plt.xlabel('Layers Resnet 50', fontsize=65)
    plt.ylabel('Layers Resnet 50', fontsize=65)
    plt.savefig('./distance/comparingModelsPetImagesNewNoPre/Case3/TeacherResNet50.png', dpi=300)
    plt.show()

    figure(figsize=(6, 6), dpi=60)
    sim_mat = np.load(PATH + 'modelY_model_Y.npy')
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('Student', fontsize=50)
    plt.xlabel('Layers Resnet 50', fontsize=65)
    plt.ylabel('Layers Resnet 50', fontsize=65)

    plt.savefig('./distance/comparingModelsPetImagesNewNoPre/Case3/StudentResnet50.png', dpi=300)
    plt.show()

    figure(figsize=(12, 12), dpi=60)
    sim_mat = np.load(PATH + 'modelX_model_Y.npy')
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('Student vs Teach', fontsize=75)
    plt.ylabel('Layers Teacher', fontsize=65)
    plt.xlabel('Layers Student', fontsize=65)
    plt.savefig('./distance/comparingModelsPetImagesNewNoPre/Case3/TeacherStd.png', dpi=300)
    plt.show()