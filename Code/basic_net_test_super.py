import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import get_superclasses, exclude_classes, to_latex_heatmap, \
    ClassSpecificImageFolderNotAlphabetic, get_classes, accuracy_superclasses, sparse2coarse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import random


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dir = "..//..//cifar//test//"
    batch_size = 128

    model_name = "..//..//Models//New_020622//resnet_coarse_lr0001_wd01_1on16.pth"

    latex = False
    plot_cf = True

    superclasses = get_superclasses()
    classes = get_classes()
    n_classes = len(classes[0])
    n_superclasses = len(superclasses)
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)
    classes.append(excluded)

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_dataset = ClassSpecificImageFolderNotAlphabetic(test_dir, all_dropped_classes=classes, transform=transform)
    classes = classes[0]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_sizes = len(test_loader)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features=n_superclasses)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()

    # evaluate_regularization(model)

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = torch.from_numpy(sparse2coarse(labels.numpy(), np.asarray(coarse_labels)))
        labels = labels.type(torch.int64)
        labels = labels.to(device)

        output = model(inputs)

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    print(classification_report(y_true, y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    # CLASSES
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(superclasses), index=[i for i in superclasses],
                         columns=[i for i in superclasses])

    if latex:
        print(to_latex_heatmap(len(superclasses), superclasses,
                               (cf_matrix / np.sum(cf_matrix) * len(superclasses)) * 100))

    if plot_cf:
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0) #vmax=1)
        plt.savefig("..\\ConfusionMatrixes\\{}-CMpercentageSuperOnly.png".format(model_name.split("\\")[-1].split(".")[0]))


