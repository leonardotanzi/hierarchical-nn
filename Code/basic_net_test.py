import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from inout import to_latex_heatmap, save_list
from evaluation import accuracy_superclasses
from utils import get_superclasses, get_classes, sparser2coarser, get_medium_labels
from dataset import exclude_classes, ImageFolderNotAlphabetic
from visualization import plot_graph_top3superclasses, plot_graph, plot_variance

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import random
import plotly.express as px


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dir = "..//..//cifar//test//"
    batch_size = 128

    model_name = "..//..//Models//New_020622//resnetfreezed_hloss_reg_lr0001_wd01_1on16.pth"

    latex = False
    plot_cf = True

    superclasses = get_superclasses()
    classes = get_classes()
    # random.seed(0)
    # random.shuffle(classes[0])
    n_classes = len(classes)
    n_superclasses = len(superclasses)
    medium_labels = get_medium_labels(superclasses)

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_dataset = ImageFolderNotAlphabetic(test_dir, classes=classes, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_sizes = len(test_loader)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features=n_classes)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()

    # evaluate_regularization(model)

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    y_pred = sparser2coarser(y_pred, np.asarray(medium_labels))

    ###############################################################################################################

    # 1) Plot Graphs
    # save_list("pkl//HLoss.pkl", y_pred)
    # plot_graph(y_pred, y_true, classes)
    # plot_graph_top3superclasses(y_pred, y_true, classes, superclasses)

    ###############################################################################################################

    # 2) Confusion Matrixes

    # 2.1) CLASSES
    print(classification_report(y_true, y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(classes), index=[i for i in classes],
                         columns=[i for i in classes])

    if latex:
        print(to_latex_heatmap(len(classes), classes,
                               (cf_matrix / np.sum(cf_matrix) * len(classes)) * 100))

    if plot_cf:
        plt.figure(figsize=(48, 28))
        sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0)
        plt.savefig("..\\ConfusionMatrixes\\{}-CM.png".format(model_name.split("//")[-1].split(".")[0]))


    # 2.2) SUPERCLASSES
    _, medium_labels = exclude_classes(superclasses_names=superclasses)
    y_true_super = sparser2coarser(y_true, np.asarray(medium_labels))
    y_pred_super = sparser2coarser(y_pred, np.asarray(medium_labels))
    print(classification_report(y_true_super, y_pred_super))
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true_super, y_pred_super)
    print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(superclasses), index=[i for i in superclasses],
                         columns=[i for i in superclasses])

    if latex:
        print(to_latex_heatmap(len(superclasses), superclasses,
                               (cf_matrix / np.sum(cf_matrix) * len(superclasses)) * 100))

    if plot_cf:
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0) #vmax=1)
        plt.savefig("..\\ConfusionMatrixes\\{}-SuperCM.png".format(model_name.split("//")[-1].split(".")[0]))


