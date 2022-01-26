import torch
import torchvision
import torch.nn as nn
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
import copy
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from utils import ClassSpecificImageFolder, exclude_classes
from CNN_CIFAR100 import ConvNet


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_size = 32
    batch_size = 128

    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dir = "..//..//cifar//test//"

    excluded, coarse_labels = exclude_classes(n_superclasses=3)
    test_dataset = ClassSpecificImageFolder(test_dir, dropped_classes=excluded, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_names = test_dataset.classes

    best_model_wts = "..\\..\\cnn_hierarchical.pth"

    model = ConvNet(num_classes=len(class_names))

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
    plt.savefig("output.png")