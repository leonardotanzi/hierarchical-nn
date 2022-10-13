import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torchvision import datasets

from evaluation import accuracy_coarser_classes, hierarchical_error
from losses import hierarchical_cc_treebased
from dataset import train_val_dataset, ImageFolderNotAlphabetic
from utils import decimal_to_string
from tree import get_tree_from_file, get_all_labels_downtop, get_all_labels_topdown, return_matrixes_downtop, return_matrixes_topdown

import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import timeit
from anytree import LevelOrderGroupIter
import random
from transformers import ViTForImageClassification, ViTFeatureExtractor
import cv2
import glob

def build_dataset(train_path, classes_list):

    savedir = "..\\bones224"
    for classes_name in classes_list:
        for c in classes_list:
            if os.path.isdir(os.path.join(train_path, c)):
                for file_name in glob.glob(os.path.join(train_path, c) + "//*.png"):
                    image = cv2.imread(file_name, cv2.COLOR_GRAY2RGB)

                    if len(image.shape) < 3:
                        image = np.stack((image,) * 3, axis=-1)
                    else:
                        print(image.shape)
                        print(file_name)

                    image = cv2.resize(image, (224, 224))
                    name = file_name.split('\\')[-1]
                    out_path = f"{savedir}\\test\\{c}\\{name}"
                    cv2.imwrite(out_path, image)
    return


if __name__ == "__main__":


    build_dataset("..//..//Dataset//bones1channel//test//", ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    n_epochs = 30
    learning_rate = 0.001
    scheduler_step_size = 40
    validation_split = 0.1

    hierarchical_loss = True
    regularization = hierarchical_loss

    architecture = "vit"
    dataset = "bones3channels"  # fgvc, cifar, imagenet, bones

    dict_architectures = {"inception": 299, "resnet": 224, "vit": 224}
    image_size = dict_architectures[architecture]

    name = f"{architecture}-{dataset}-extractor"

    run_scheduler = False
    sp_regularization = False
    weight_decay = 0.1
    less_samples = True
    reduction_factor = 1 if less_samples is False else 1
    freeze = False
    load = False

    # Classes and superclasses
    tree_file = f"..//..//Dataset//{dataset}//tree.txt"
    tree = get_tree_from_file(tree_file)
    all_leaves = [leaf.name for leaf in tree.leaves]

    all_nodes_names = [[node.name for node in children] for children in LevelOrderGroupIter(tree)][1:]
    all_nodes = [[node for node in children] for children in LevelOrderGroupIter(tree)][1:]
    # to convert the fine labels to any other level, read each node, count the leaves and add one integer for each
    # leaves in the node
    all_labels_topdown = get_all_labels_topdown(tree)
    all_labels_downtop = get_all_labels_downtop(tree)
    all_labels = [*all_labels_topdown, *all_labels_downtop]

    # all_labels = get_all_labels_topdown(tree)

    matrixes_topdown = return_matrixes_topdown(tree, plot=False)
    matrixes_downtop = return_matrixes_downtop(tree, plot=False)
    matrixes = [*matrixes_topdown, *matrixes_downtop]

    # matrixes = return_matrixes_topdown(tree, plot=False)

    lens = [len(n) for n in all_nodes]

    # Path
    model_path = "..//..//Models//Mat_version_210622//"
    if hierarchical_loss and not regularization:
        model_name = os.path.join(model_path,
                                  f"{name}_hloss_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    elif regularization and not hierarchical_loss:
        model_name = os.path.join(model_path,
                                  f"{name}_reg_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    elif regularization and hierarchical_loss:
        model_name = os.path.join(model_path,
                                  f"{name}_hloss_reg_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    else:
        model_name = os.path.join(model_path,
                                  f"{name}_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    print(f"Model name: {model_name}")

    # Log
    writer = SummaryWriter(os.path.join("..//Logs//Mat_version_210622//", model_name.split("//")[-1].split(".")[0]))

    # Dataset
    train_dir = f"..//..//Dataset//{dataset}//train//"
    test_dir = f"..//..//Dataset//{dataset}//test//"

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((image_size, image_size))])

    train_dataset = ImageFolderNotAlphabetic(train_dir, classes=all_leaves, transform=transform)

    dataset = train_val_dataset(train_dataset, validation_split, reduction_factor, reduce_val=False, reduction_factor_val=32)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    print(f"Using {dataset_sizes['train']} samples for training, {dataset_sizes['train']/len(all_leaves)} for each class")

    # Check lr_ratio
    lr_ratio = 1 / len(train_loader)
    print(f"LR should be around {lr_ratio:.4f}")

    # Model
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

        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, out_features=len(all_leaves))
    # Freeze layers
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.to(device)

    if load:
        model.load_state_dict(torch.load("..//..//Models//Mat_version_210622//inception_fgvc_hloss_reg_lr0001_wd01_1on1_best.pth"))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if regularization \
        else torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler
    if run_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.3)
        # every n=step_size epoch the lr is multiplied by gamma

    # Parameters for training
    n_total_steps_train = len(train_loader)
    n_total_steps_val = len(val_loader)
    platoon = 0
    best_acc = 0.0
    associated_medium_acc = 0.0
    associated_coarse_acc = 0.0

    for epoch in range(n_epochs):
        start = timeit.default_timer()
        print("-" * 200)
        print(f"Epoch {epoch + 1}/{n_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
                n_total_steps = n_total_steps_train
            else:
                model.eval()
                loader = val_loader
                n_total_steps = n_total_steps_val

            running_loss = 0.0
            running_corrects = 0
            running_corrects_coarser_level = [0 for i in range(len(all_labels))]
            running_loss_fine = 0.0
            running_loss_coarser_level = [0.0 for i in range(len(all_labels))]

            epoch_acc_coarser = [0 for i in range(len(all_labels))]
            epoch_loss_coarser = [0.0 for i in range(len(all_labels))]

            # Iterate over data
            for j, (inputs, labels) in enumerate(loader):
                inputs = inputs
                labels = labels.to(device)

                # Forward
                with torch.set_grad_enabled(phase == "train"):

                    if architecture == "vit":
                        inputs = feature_extractor(images=inputs, return_tensors="pt")
                        outputs = model(**inputs).logits
                    else:
                        outputs = model(inputs)

                    # x = hierarchical_accuracy(outputs, labels, tree, all_leaves, device)

                    _, preds = torch.max(outputs, 1)

                    loss, loss_dict = hierarchical_cc_treebased(outputs, labels, tree, lens, all_labels, all_leaves,
                                                                model, 0.0, device, hierarchical_loss, regularization,
                                                                sp_regularization, weight_decay, matrixes, architecture)

                    # Backward + optimize
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
                running_loss_fine += loss_dict["loss_fine"]

                for i in range(len(all_labels)):
                    running_corrects_coarser_level[i] += accuracy_coarser_classes(outputs, labels, np.asarray(all_labels[i]),
                                                                   len(all_labels[i]), device)
                    running_loss_coarser_level[i] += loss_dict[f"loss_{i}"]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_loss_fine = running_loss_fine / dataset_sizes[phase]

            for i in range(len(all_labels)):
                epoch_acc_coarser[i] = running_corrects_coarser_level[i] / dataset_sizes[phase]
                epoch_loss_coarser[i] = running_loss_coarser_level[i] / dataset_sizes[phase]

            print(f"Step {j + 1}/{n_total_steps}, {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            for i in range(len(all_labels)):
                print(f"{phase} Loss {i}: {epoch_loss_coarser[i]:.4f}, Accuracy {i}: {epoch_acc_coarser[i]:.4f}")

            if (j + 1) % n_total_steps == 0:
                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        platoon = 0
                        best_model_name = model_name[:-4] + "_best.pth"
                        torch.save(model.state_dict(), best_model_name)
                        print(f"New best accuracy {best_acc:.4f}, saving best model")

                    if epoch_acc < best_acc:
                        platoon += 1
                        print(f"{platoon} epochs without improvement, best accuracy: {best_acc:.4f}")

                    print("End of validation epoch.")
                    writer.add_scalar("Validation loss", epoch_loss, epoch)
                    writer.add_scalar("Validation accuracy", epoch_acc, epoch)
                    for i in range(len(all_labels)):
                        writer.add_scalar(f"Validation accuracy {i}", epoch_acc_coarser[i], epoch)

                    writer.add_scalars("Training vs. validation loss",
                                       {"Training": epoch_loss_compare, "Validation": epoch_loss}, epoch)
                    writer.add_scalars("Training vs. validation accuracy",
                                       {"Training": epoch_acc_compare, "Validation": epoch_acc}, epoch)

                    plot_dict = {"Loss": epoch_loss, "Fine Loss": epoch_loss_fine}

                    for i in range(len(all_labels)):
                        plot_dict[f"Loss {i}"] = epoch_loss_coarser[i]

                    writer.add_scalars("Losses and penalties validation", plot_dict, epoch)

                elif phase == "train":
                    if run_scheduler:
                        scheduler.step()
                    print("End of training epoch.")
                    writer.add_scalar("Training loss", epoch_loss, epoch)
                    writer.add_scalar("Training accuracy", epoch_acc, epoch)
                    for i in range(len(all_labels)):
                        writer.add_scalar(f"Training accuracy {i}", epoch_acc_coarser[i], epoch)

                    plot_dict = {"Loss": epoch_loss, "Fine Loss": epoch_loss_fine}

                    for i in range(len(all_labels)):
                        plot_dict[f"Loss {i}"] = epoch_loss_coarser[i]

                    writer.add_scalars("Losses and penalties training", plot_dict, epoch)
                    epoch_loss_compare = epoch_loss
                    epoch_acc_compare = epoch_acc

        stop = timeit.default_timer()
        print(f"Elapsed time {stop-start:.4f}")

    last_model_name = model_name[:-4] + "_last.pth"

    writer.flush()
    writer.close()
