import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import get_superclasses, exclude_classes, to_latex_heatmap, \
    ClassSpecificImageFolder, get_classes, accuracy_superclasses, sparse2coarse
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

    data_dir = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'

    batch_size = 64
    image_size = 224

    model_name = "..//..//Models//Final_100522//bonesfreezed_lr0001_wd001_1on1.pth"

    latex = False
    plot_cf = False

    transform = Compose([ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                         Resize(size=(image_size, image_size))])
    test_dataset = ClassSpecificImageFolder(data_dir, dropped_classes=["A", "B", "Broken", "TestDoctors"], transform=transform)

    # prepare superclasses
    coarse_classes = ["Broken", "Unbroken"]
    medium_classes = ["A", "B", "Unbroken"]
    fine_classes = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]

    n_classes = len(fine_classes)

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

    print(classification_report(y_true, y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    # # CLASSES
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(classes), index=[i for i in classes],
    #                      columns=[i for i in classes])
    #
    # if latex:
    #     print(to_latex_heatmap(len(classes), classes,
    #                            (cf_matrix / np.sum(cf_matrix) * len(classes)) * 100))
    #
    # if plot_cf:
    #     plt.figure(figsize=(12, 7))
    #     sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0, vmax=1)
    #     plt.savefig("..\\ConfusionMatrixes\\{}-CMpercentage.png".format(model_name.split("\\")[-1].split(".")[0]))
    #
    # # SUPERCLASSES
    # _, coarse_labels = exclude_classes(superclasses_names=superclasses)
    # y_true_super = sparse2coarse(y_true, np.asarray(coarse_labels))
    # y_pred_super = sparse2coarse(y_pred, np.asarray(coarse_labels))
    # print(classification_report(y_true_super, y_pred_super))
    # # Build confusion matrix
    # cf_matrix = confusion_matrix(y_true_super, y_pred_super)
    # print(cf_matrix)
    #
    # # CLASSES
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(superclasses), index=[i for i in superclasses],
    #                      columns=[i for i in superclasses])
    #
    # if latex:
    #     print(to_latex_heatmap(len(superclasses), superclasses,
    #                            (cf_matrix / np.sum(cf_matrix) * len(superclasses)) * 100))
    #
    # if plot_cf:
    #     plt.figure(figsize=(12, 7))
    #     sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0, vmax=1)
    #     plt.savefig("..\\ConfusionMatrixes\\{}-CMpercentage2.png".format(model_name.split("\\")[-1].split(".")[0]))
    #

