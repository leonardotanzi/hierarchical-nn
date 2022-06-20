import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from evaluation import accuracy_coarser_classes
from losses import hierarchical_cc_3levels, hierarchical_cc_treebased
from dataset import train_val_dataset, ImageFolderNotAlphabetic
from utils import get_superclasses, get_classes, get_hyperclasses, decimal_to_string,  get_medium_labels, get_coarse_labels
from tree import get_tree_CIFAR

import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import timeit
from anytree import LevelOrderGroupIter
import random


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    n_epochs = 100
    learning_rate = 0.001
    scheduler_step_size = 40
    validation_split = 0.1

    hierarchical_loss = True
    medium_loss = True
    coarse_loss = True
    regularization = False
    name = "cc"

    run_scheduler = False
    sp_regularization = False
    weight_decay = 0.1
    less_samples = True
    reduction_factor = 1 if less_samples is False else 16

    # Classes and superclasses
    tree = get_tree_CIFAR()
    all_leaves = [leaf.name for leaf in tree.leaves]
    all_nodes_names = [[node.name for node in children] for children in LevelOrderGroupIter(tree)][1:]
    all_nodes = [[node for node in children] for children in LevelOrderGroupIter(tree)][1:]
    all_labels = []
    # to convert the fine labels to any other level, read each node, count the leaves and add one integer for each
    # leaves in the node
    for node in all_nodes[:-1]:
        labels = []
        for i, single_node in enumerate(node):
            for j in range(len(single_node.leaves)):
                labels.append(i)
        all_labels.append(labels)

    # fine_classes = all_nodes[0]
    # medium_classes = all_nodes[1]
    # coarse_classes = all_nodes[2]
    lens = [len(n) for n in all_nodes]
    # n_fine_classes = len(fine_classes)
    # n_medium_classes = len(medium_classes)
    # n_coarse_classes = len(coarse_classes)

    # medium_labels = get_medium_labels(superclasses_names=medium_classes)
    # coarse_labels = get_coarse_labels(superclasses_names=coarse_classes)

    # Path
    model_path = "..//..//Models//New_160622//"
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
    writer = SummaryWriter(os.path.join("..//Logs//New_160622//", model_name.split("//")[-1].split(".")[0]))

    # Dataset
    train_dir = "..//..//cifar//train//"
    test_dir = "..//..//cifar//test//"

    tree = get_tree_CIFAR()

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = ImageFolderNotAlphabetic(train_dir, classes=all_leaves, transform=transform)

    dataset = train_val_dataset(train_dataset, validation_split, reduction_factor)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4,)
    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    # Check lr_ratio
    lr_ratio = 1 / len(train_loader)
    print(f"LR should be around {lr_ratio:.4f}")

    # Model
    model = models.resnet18(pretrained=True)
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Add last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features=len(all_leaves))
    model.to(device)

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
            running_corrects_coarser_level = [0 for i in range(len(all_nodes)-1)]
            running_loss_fine = 0.0
            running_loss_coarser_level = [0.0 for i in range(len(all_nodes)-1)]

            epoch_acc_coarser = [0 for i in range(len(all_nodes)-1)]
            epoch_loss_coarser = [0.0 for i in range(len(all_nodes)-1)]

            # Iterate over data
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss, loss_dict = hierarchical_cc_treebased(outputs, labels, tree, lens, all_labels, model, 0.0, device,
                                                    hierarchical_loss, regularization, sp_regularization, weight_decay)

                    # Backward + optimize
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
                running_loss_fine += loss_dict["loss_fine"]

                for i in range(len(all_nodes)-1):
                    running_corrects_coarser_level[i] += accuracy_coarser_classes(outputs, labels, np.asarray(all_labels[i]),
                                                                   len(all_labels[i]), device)
                    running_loss_coarser_level[i] += loss_dict[f"loss_{i}"]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_loss_fine = running_loss_fine / dataset_sizes[phase]

            for i in range(len(all_nodes)-1):
                epoch_acc_coarser[i] = running_corrects_coarser_level[i] / dataset_sizes[phase]
                epoch_loss_coarser[i] = running_loss_coarser_level[i] / dataset_sizes[phase]

            print(f"Step {i + 1}/{n_total_steps}, {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            for i in range(len(all_nodes) - 1):
                print(f"Loss {i}: {epoch_loss_coarser[i]:.4f}, Accuracy {i}: {epoch_acc_coarser[i]:.4f}")

            if (i + 1) % n_total_steps == 0:
                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        # associated_medium_acc = epoch_acc_medium
                        # associated_coarse_acc = epoch_acc_coarse
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
                    for i in range(len(all_nodes) - 1):
                        writer.add_scalar(f"Validation accuracy {i}", epoch_acc_coarser[i], epoch)

                    writer.add_scalars("Training vs. validation loss",
                                       {"Training": epoch_loss_compare, "Validation": epoch_loss}, epoch)
                    writer.add_scalars("Training vs. validation accuracy",
                                       {"Training": epoch_acc_compare, "Validation": epoch_acc}, epoch)

                    plot_dict = {"Loss": epoch_loss, "Fine Loss": epoch_loss_fine}

                    for i in range(len(all_nodes) - 1):
                        plot_dict[f"Loss {i}"] = epoch_loss_coarser[i]

                    writer.add_scalars("Losses and penalties validation", plot_dict, epoch)

                elif phase == "train":
                    if run_scheduler:
                        scheduler.step()
                    print("End of training epoch.")
                    writer.add_scalar("Training loss", epoch_loss, epoch)
                    writer.add_scalar("Training accuracy", epoch_acc, epoch)
                    for i in range(len(all_nodes) - 1):
                        writer.add_scalar(f"Training accuracy {i}", epoch_acc_coarser[i], epoch)

                    plot_dict = {"Loss": epoch_loss, "Fine Loss": epoch_loss_fine}

                    for i in range(len(all_nodes) - 1):
                        plot_dict[f"Loss {i}"] = epoch_loss_coarser[i]

                    writer.add_scalars("Losses and penalties training", plot_dict, epoch)
                    epoch_loss_compare = epoch_loss
                    epoch_acc_compare = epoch_acc

        stop = timeit.default_timer()
        print(f"Elapsed time {stop-start:.4f}")

    last_model_name = model_name[:-4] + "_last.pth"
    torch.save(model.state_dict(), last_model_name)

    writer.flush()
    writer.close()
