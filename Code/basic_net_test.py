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


def evaluate_regularization(model, n_class=5, n_superclass=20):
    coarse_penalty = 0.0
    fine_penalty = 0.0
    for i in range(n_superclass):
        coarse_penalty += (torch.linalg.norm(
            torch.sum(model.fc.weight.data[i * n_class:i * n_class + n_class], dim=0))) ** 2
    for i in range(n_class * n_superclass):
        sc_index = 1 // 5
        fine_penalty += (torch.linalg.norm(model.fc.weight.data[i] - 1 / n_class * torch.sum(
            model.fc.weight.data[sc_index * n_class:sc_index * n_class + n_class]))) ** 2

    print(coarse_penalty)
    print(fine_penalty)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_dir = "..//..//cifar//test//"
    batch_size = 128

    latex = False
    plot_cf = False

    superclasses = get_superclasses()
    classes = get_classes()
    # given the list of superclasses, returns the class to exclude and the coarse label
    n_classes = len(classes[0])
    n_superclasses = len(superclasses)
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)
    classes.append(excluded)

    test_dataset = ClassSpecificImageFolderNotAlphabetic(test_dir, all_dropped_classes=classes,
                                                              transform=transform)

    classes = classes[0]
    model_name = "..//..//Models//Final_100522//resnet_hloss_reg_lr0001_wd01_1on1.pth"

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataset_sizes = len(test_loader)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features  # input features for the last layers
    model.fc = nn.Linear(num_ftrs, out_features=n_classes)
    model.load_state_dict(torch.load(model_name))  # it takes the loaded dictionary, not the path file itself
    model.to(device)
    model.eval()

    evaluate_regularization(model)

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

    print(classification_report(y_true, y_pred))
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    # CLASSES
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(classes), index=[i for i in classes],
                         columns=[i for i in classes])

    if latex:
        print(to_latex_heatmap(len(classes), classes,
                               (cf_matrix / np.sum(cf_matrix) * len(classes)) * 100))

    if plot_cf:
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0, vmax=1)
        plt.savefig("..\\ConfusionMatrixes\\{}-CMpercentage.png".format(model_name.split("\\")[-1].split(".")[0]))

    # SUPERCLASSES

    _, coarse_labels = exclude_classes(superclasses_names=superclasses)
    y_true_super = sparse2coarse(y_true, np.asarray(coarse_labels))
    y_pred_super = sparse2coarse(y_pred, np.asarray(coarse_labels))
    print(classification_report(y_true_super, y_pred_super))
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true_super, y_pred_super)
    print(cf_matrix)

    # CLASSES
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(superclasses), index=[i for i in superclasses],
                         columns=[i for i in superclasses])

    if latex:
        print(to_latex_heatmap(len(superclasses), superclasses,
                               (cf_matrix / np.sum(cf_matrix) * len(superclasses)) * 100))

    if plot_cf:
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0, vmax=1)
        plt.savefig("..\\ConfusionMatrixes\\{}-CMpercentage2.png".format(model_name.split("\\")[-1].split(".")[0]))


