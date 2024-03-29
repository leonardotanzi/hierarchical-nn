import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from inout import to_latex_heatmap
from evaluation import hierarchical_error
from tree import get_tree_from_file, get_all_labels_downtop, get_all_labels_topdown
from utils import seed_everything

from anytree import LevelOrderGroupIter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle
from transformers import ViTForImageClassification
import argparse


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_everything(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Inception, ResNet or ViT")
    args = vars(ap.parse_args())

    architecture = args["model"]

    batch_size = 32

    dict_architectures = {"inception": 299, "resnet": 299, "vit": 224}
    image_size = dict_architectures[architecture]

    model_name = f"..//..//Models//newpoints//{architecture}-imagenet_hloss_reg_lr0001_wd01_1on1_best.pth"

    latex = False
    plot_cf = False

    tree = get_tree_from_file("..//..//Dataset//ImageNet64//tree.txt")

    all_leaves = [leaf.name for leaf in tree.leaves]

    all_nodes_names = [[node.name for node in children] for children in LevelOrderGroupIter(tree)][1:]
    all_nodes = [[node for node in children] for children in LevelOrderGroupIter(tree)][1:]

    all_labels_topdown = get_all_labels_topdown(tree)
    all_labels_downtop = get_all_labels_downtop(tree)
    all_labels = [*all_labels_topdown, *all_labels_downtop]

    lens = [len(n) for n in all_nodes]

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((image_size, image_size))])

    pkl = "..//..//pkl//imagenet_dataset299.pkl" if architecture == "inception" or architecture == "resnet" else "..//..//pkl//imagenet_dataset.pkl"

    with open(pkl, "rb") as f:
        dataset = pickle.load(f)

    test_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

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

    # model = nn.DataParallel(model)
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
        if architecture == "vit":
            inputs = Resize(size=224)(inputs)

        labels = labels.to(device)

        if architecture == "vit":
            outputs = model(inputs).logits
        else:
            outputs = model(inputs)

        h_acc += hierarchical_error(outputs, labels, tree, all_leaves, device)

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    print(f"Hierarchical_error is {h_acc/dataset_size:.4f}")


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
