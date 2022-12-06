import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from inout import to_latex_heatmap
from utils import seed_everything
from evaluation import hierarchical_error
from dataset import ImageFolderNotAlphabetic
from tree import get_tree_from_file, get_all_labels_topdown, get_all_labels_downtop, \
    return_matrixes_topdown, return_matrixes_downtop

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import copy


def build_model(model, model_path, n_class):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features=n_class)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    architecture = "inception"
    dataset = "cifar"

    model_path = "..//..//Models//cascade//"

    dict_architectures = {"inception": 299, "resnet": 224, "vit": 224}

    image_size = dict_architectures[architecture]

    test_dir = f"..//..//Dataset//{dataset}//test//"
    tree_file = f"..//..//Dataset//{dataset}//tree_subset.txt"

    batch_size = 1

    latex = False
    plot_cf = True

    tree = get_tree_from_file(tree_file)

    all_leaves = [leaf.name for leaf in tree.leaves]

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((image_size, image_size))])

    test_dataset = ImageFolderNotAlphabetic(test_dir, classes=all_leaves, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_size = len(test_loader)

    model = models.inception_v3(pretrained=True)
    model.aux_logits = False

    models = ["resnet-cifar-seaanimal_outdoorscenes_hloss_reg_lr0001_wd01_1on8_best",
              "resnet-cifar-aquaticmammals_fish_hloss_reg_lr0001_wd01_1on8_best",
              "resnet-cifar-largemanmade_largenatural_hloss_reg_lr0001_wd01_1on8_best",
              "resnet-cifar-first_hloss_reg_lr0001_wd01_1on8_best",
              "resnet-cifar-second_hloss_reg_lr0001_wd01_1on8_best",
              "resnet-cifar-third_hloss_reg_lr0001_wd01_1on8_best",
              "resnet-cifar-fourth_hloss_reg_lr0001_wd01_1on8_best"]

    model1 = build_model(copy.deepcopy(model), model_path + models[0] + ".pth", n_class=2)

    model11 = build_model(copy.deepcopy(model), model_path + models[1] + ".pth", n_class=2)
    model12 = build_model(copy.deepcopy(model), model_path + models[2] + ".pth", n_class=2)

    model111 = build_model(copy.deepcopy(model), model_path + models[3] + ".pth", n_class=5)
    model112 = build_model(copy.deepcopy(model), model_path + models[4] + ".pth", n_class=5)
    model121 = build_model(copy.deepcopy(model), model_path + models[5] + ".pth", n_class=5)
    model122 = build_model(copy.deepcopy(model), model_path + models[6] + ".pth", n_class=5)

    y_pred = []
    y_true = []

    # iterate over test data
    h_err = 0.0

    precise_pred = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        pred = model1(inputs)
        first_pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()

        if first_pred == 0:
            pred = model11(inputs)
            second_pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()

            if second_pred == 0:
                pred = model111(inputs)
                third_pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
                if third_pred == 0:
                    precise_pred.append(all_leaves.index("beaver"))
                elif third_pred == 1:
                    precise_pred.append(all_leaves.index("dolphin"))
                elif third_pred == 2:
                    precise_pred.append(all_leaves.index("otter"))
                elif third_pred == 3:
                    precise_pred.append(all_leaves.index("seal"))
                elif third_pred == 4:
                    precise_pred.append(all_leaves.index("whale"))
            elif second_pred == 1:
                pred = model112(inputs)
                third_pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
                if third_pred == 0:
                    precise_pred.append(all_leaves.index("aquarium_fish"))
                elif third_pred == 1:
                    precise_pred.append(all_leaves.index("flatfish"))
                elif third_pred == 2:
                    precise_pred.append(all_leaves.index("ray"))
                elif third_pred == 3:
                    precise_pred.append(all_leaves.index("shark"))
                elif third_pred == 4:
                    precise_pred.append(all_leaves.index("trout"))

        elif first_pred == 1:
            pred = model12(inputs)
            second_pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()

            if second_pred == 0:
                pred = model121(inputs)
                third_pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
                if third_pred == 0:
                    precise_pred.append(all_leaves.index("bridge"))
                elif third_pred == 1:
                    precise_pred.append(all_leaves.index("castle"))
                elif third_pred == 2:
                    precise_pred.append(all_leaves.index("house"))
                elif third_pred == 3:
                    precise_pred.append(all_leaves.index("road"))
                elif third_pred == 4:
                    precise_pred.append(all_leaves.index("skyscraper"))
            elif second_pred == 1:
                pred = model122(inputs)
                third_pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
                if third_pred == 0:
                    precise_pred.append(all_leaves.index("cloud"))
                elif third_pred == 1:
                    precise_pred.append(all_leaves.index("forest"))
                elif third_pred == 2:
                    precise_pred.append(all_leaves.index("mountain"))
                elif third_pred == 3:
                    precise_pred.append(all_leaves.index("plain"))
                elif third_pred == 4:
                    precise_pred.append(all_leaves.index("sea"))

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    print(f"Hierarchical error is {h_err/dataset_size:.4f}")

    # 2) Confusion Matrixes

    # 2.1) CLASSES
    print(classification_report(y_true, precise_pred))
    cf_matrix = confusion_matrix(y_true, precise_pred)
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


