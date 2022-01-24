import torch
import torchvision
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torchsummary import summary
from torch.optim import SGD
from torchvision import datasets, models
import os
import os
import copy
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from myCNN import ClassSpecificImageFolder


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

if __name__ == "__main__":

    image_size = 224

    transform = Compose([ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                         Resize(size=(image_size, image_size))])

    test_dir = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'
    test_dataset = ClassSpecificImageFolder(test_dir, dropped_classes=["A", "B", "Broken", "TestDoctors"],
                                            transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class_names = test_dataset.classes

    best_model_wts = "cnn_hierarchical.pth"
    device = "cuda:0"
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features  # input features for the last layers
    model.fc = nn.Linear(num_ftrs, out_features=len(class_names))  # we have 2 classes now

    model.to(device)

    model.load_state_dict(torch.load(best_model_wts))

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')