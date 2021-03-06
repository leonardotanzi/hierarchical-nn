import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torchvision

from inout import to_latex_heatmap, save_list
from evaluation import hierarchical_accuracy
from dataset import exclude_classes, ImageFolderNotAlphabetic
from visualization import plot_graph_top3superclasses, plot_graph, plot_variance
from tree import get_tree_CIFAR, get_all_labels_topdown, get_all_labels_downtop, return_matrixes_topdown, return_matrixes_downtop

from anytree import LevelOrderGroupIter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import random
import plotly.express as px
from transformers import ViTForImageClassification


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dir = "..//..//Dataset//cifar//test//"
    batch_size = 32

    architecture = "vit"

    model_name = "..//..//Models//Mat_version_210622//vit_cifar100_hloss_reg_lr0001_wd01_1on8_best.pth"

    latex = False
    plot_cf = True

    tree = get_tree_CIFAR()
    all_leaves = [leaf.name for leaf in tree.leaves]

    all_nodes_names = [[node.name for node in children] for children in LevelOrderGroupIter(tree)][1:]
    all_nodes = [[node for node in children] for children in LevelOrderGroupIter(tree)][1:]

    all_labels_topdown = get_all_labels_topdown(tree)
    all_labels_downtop = get_all_labels_downtop(tree)
    all_labels = [*all_labels_topdown, *all_labels_downtop]

    matrixes_topdown = return_matrixes_topdown(tree, plot=False)
    matrixes_downtop = return_matrixes_downtop(tree, plot=False)
    matrixes = [*matrixes_topdown, *matrixes_downtop]

    lens = [len(n) for n in all_nodes]

    if architecture == "inceptionv3":
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((299, 299))])
    elif architecture == "resnet18":
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((224, 224))])

    test_dataset = ImageFolderNotAlphabetic(test_dir, classes=all_leaves, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_size = len(test_loader)

    if architecture == "inception":
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
    elif architecture == "resnet50":
        model = models.resnet50(pretrained=True)
    elif architecture == "resnet18":
        model = models.resnet18(pretrained=True)
    elif architecture == "vit":
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    if architecture == "vit":
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, out_features=len(all_leaves))
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_features=len(all_leaves))

    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()

    # evaluate_regularization(model)

    y_pred = []
    y_true = []

    # iterate over test data
    h_acc = 0.0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if architecture == "vit":
            outputs = model(inputs).logits
        else:
            outputs = model(inputs)

        h_acc += hierarchical_accuracy(outputs, labels, tree, all_leaves, device)

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    print(f"Hierarchical_accuracy is {h_acc/dataset_size:.4f}")

    ###############################################################################################################

    # 1) Plot Graphs
    # y_pred = sparser2coarser(y_pred, np.asarray(medium_labels))
    # save_list("pkl//HLoss.pkl", y_pred)
    # plot_graph(y_pred, y_true, classes)
    # plot_graph_top3superclasses(y_pred, y_true, classes, superclasses)

    ###############################################################################################################

    # 2) Confusion Matrixes

    # 2.1) CLASSES
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



