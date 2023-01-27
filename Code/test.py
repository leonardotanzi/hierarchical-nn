import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from inout import to_latex_heatmap
from utils import seed_everything
from evaluation import hierarchical_error, cpb
from dataset import ImageFolderNotAlphabetic
from tree import get_tree_from_file, get_all_labels_topdown, get_all_labels_downtop, \
    return_matrixes_topdown, return_matrixes_downtop

from anytree import LevelOrderGroupIter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from transformers import ViTForImageClassification
import argparse


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Inception, ResNet or ViT")
    ap.add_argument("-d", "--dataset", required=True, help="fgvc, cifar, bones")
    args = vars(ap.parse_args())

    architecture = args["model"]
    dataset_name = args["dataset"]

    model_name = f"..//..//Models//Mat_version_210622//{architecture}-{dataset_name}//{architecture}-{dataset_name}_hloss_reg_lr0001_wd01_1on1_best.pth"

    dict_architectures = {"inception": 299, "resnet": 224, "vit": 224}

    image_size = dict_architectures[architecture]

    test_dir = f"..//..//Dataset//{dataset_name}//test//"

    batch_size = 32

    latex = False
    plot_cf = False

    tree_file = f"..//..//Dataset//{dataset_name}//tree.txt"
    tree = get_tree_from_file(tree_file)

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

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((image_size, image_size))])

    test_dataset = ImageFolderNotAlphabetic(test_dir, classes=all_leaves, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_size = len(test_loader)

    if architecture == "inception":
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_features=len(all_leaves))
    elif architecture == "resnet":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_features=len(all_leaves))
    elif architecture == "vit":
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, out_features=len(all_leaves))

    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []

    # iterate over test data
    h_err = 0.0
    # cpb_val = 0.0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).logits if architecture == "vit" else model(inputs)

        h_err += hierarchical_error(outputs, labels, tree, all_leaves, device)
        # cpb_val += cpb(outputs, labels, tree, all_leaves, device)
        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    print(f"Hierarchical error is {h_err/dataset_size:.4f}")
    # print(f"CPB is {cpb_val/dataset_size:.4f}")
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
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f", vmin=0)
        plt.savefig("..\\ConfusionMatrixes\\{}-CM.png".format(model_name.split("//")[-1].split(".")[0]))


