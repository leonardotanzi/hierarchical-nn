import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
import torchvision

from inout import to_latex_heatmap, save_list
from evaluation import accuracy_coarser_classes, hierarchical_accuracy, fairness_gender
from dataset import ImageFolderNotAlphabetic

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import random
import plotly.express as px
import pickle
import glob
import os
import cv2
import pandas


class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, classes, transform=None):

        df = pandas.read_csv(f"..//..//dataset//FairFace//fairface_label_val.csv")

        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        file_list = glob.glob(self.root_dir + "*")
        print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "\\*.jpg"):
                key = "val/" + img_path.split("\\")[-1]
                race = df[df['file'] == key].race.values[0]
                race = race.replace(" ", "")
                race = race.replace("_", "")
                self.data.append([img_path, class_name, race])

        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name, race = self.data[idx]
        img = cv2.imread(img_path)

        class_id = self.classes.index(class_name)

        if self.transform:
            img = self.transform(img)

        class_id = torch.tensor([class_id])

        return img, class_name, race


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128

    model_name = "..\\..\\Models\\Mat_version_210622\\resnet-fairface-flat_lr0001_wd01_1on1_best.pth"

    latex = False
    plot_cf = True

    classes = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-99"]

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    val_dir = "..\\..\\Dataset\\FairFace\\FairFace_flat\\val\\"

    # Load the data: train and test sets
    # val_dataset = ImageFolderNotAlphabetic(val_dir, classes=classes, transform=transform)
    # test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    val_custom = CustomDataset(val_dir, classes, transform)
    test_loader_custom = DataLoader(val_custom, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    dataset_size = len(test_loader_custom)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features=len(classes))
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()

    # evaluate_regularization(model)

    y_pred = []
    y_true = []

    # iterate over test data
    h_acc = 0.0

    for inputs, labels, race in test_loader_custom:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)




    ###############################################################################################################

    # 1) Plot Graphs
    # y_pred = sparser2coarser(y_pred, np.asarray(medium_labels))
    # save_list("pkl//HLoss.pkl", y_pred)
    # plot_graph(y_pred, y_true, classes)
    # plot_graph_top3superclasses(y_pred, y_true, classes, superclasses)

    ###############################################################################################################

    # 2) Confusion Matrixes

    # 2.1) CLASSES
    print(f"Fairness metric: {fairness_gender(y_true, y_pred, 'f1-score', all_leaves)}")
    print(classification_report(y_true, y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(all_leaves), index=[i for i in all_leaves],
                         columns=[i for i in all_leaves])

    if latex:
        print(to_latex_heatmap(len(all_leaves), all_leaves,
                               (cf_matrix / np.sum(cf_matrix) * len(all_leaves)) * 100))

    if plot_cf:
        plt.figure(figsize=(48, 28))
        sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0)
        plt.savefig("..\\ConfusionMatrixes\\{}-CM.png".format(model_name.split("//")[-1].split(".")[0]))



