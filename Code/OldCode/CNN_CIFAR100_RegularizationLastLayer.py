from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
import torch.nn.functional as F
import os
from utils import class_specific_image_folder_not_alphabetic, train_val_dataset, sparse2coarse, exclude_classes, \
    get_classes, get_superclasses, accuracy_superclasses, hierarchical_cc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from torchvision import models
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


# def hierarchical_cc(predicted, actual, coarse_labels, n_class, n_superclass, model, device, hierarchical_loss=True, regularization=True, weight_decay1=None, weight_decay2=None):
#
#     batch = predicted.size(0)
#
#     # compute the loss for fine classes
#     loss_fine = F.cross_entropy(predicted, actual, reduction="mean")
#
#     # define an empty vector which contains 20 superclasses prediction for each samples
#     predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device="cuda:0")
#
#     for k in range(n_superclass):
#         # obtain the indexes of the superclass number k
#         indexes = list(np.where(coarse_labels == k))[0]
#         # for each index, sum all the probability related to that superclass
#         for j in indexes:
#             predicted_coarse[:, k] = predicted_coarse[:, k] + predicted[:, j]
#
#     actual_coarse = sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)
#
#     loss_coarse = F.cross_entropy(predicted_coarse, torch.from_numpy(actual_coarse).type(torch.int64).to(device), reduction="mean")
#
#     # I take all the w_classes related to the index. so for example a sample which lable is 2 will get the regularizer
#     # in position 2 associated, the same for superclasses
#
#     # if weight_decay1 is not None:
#     #     regularizer1 = w_classes[actual]
#     #     regularizer2 = w_superclasses[torch.from_numpy(actual_coarse).type(torch.int64)]
#     #
#     #     # return loss_fine + loss_coarse + weight_decay1 * torch.linalg.norm(regularizer1) + weight_decay2 * torch.linalg.norm(regularizer2)
#     #     return loss_fine + weight_decay1 * torch.linalg.norm(regularizer1) + weight_decay2 * torch.linalg.norm(regularizer2)
#
#     return loss_fine + loss_coarse


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dir = "../../../cifar/train//"
    test_dir = "../../../cifar/test//"

    num_epochs = 200
    batch_size = 128
    learning_rate = 0.01
    early_stopping = 400

    hierarchical_loss = False
    regularization = False
    weight_decay = 0.1
    all_superclasses = True
    less_samples = True
    reduction_factor = 2
    freeze = True

    if hierarchical_loss and not regularization:
        model_name = "..//..//New_210322//resnet_hloss_1on{}.pth".format(reduction_factor)
    elif regularization and not hierarchical_loss:
        model_name = "..//..//New_210322//resnet_reg_1on{}.pth".format(reduction_factor)
    elif regularization and hierarchical_loss:
        model_name = "..//..//New_210322//resnet_hloss_reg_1on{}.pth".format(reduction_factor)
    else:
        model_name = "..//..//New_210322//resnet_1on{}.pth".format(reduction_factor)

    model_name = "..//..//Models//New_210322//adam_lr001.pth" #resnet_1on{}_wd01.pth".format(reduction_factor)
    writer = SummaryWriter(os.path.join("..//Logs//New_210322//", model_name.split("//")[-1].split(".")[0]))

    classes_name = get_classes()

    if not all_superclasses:
        # read superclasses, you can manually select some or get all with get_superclasses()
        superclasses = ["flowers", "fruit and vegetables", "trees"]
    else:
        superclasses = get_superclasses()

    # given the list of superclasses, returns the class to exclude and the coarse label
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)

    classes_name.append(excluded)

    # take as input a list of list with the first element being the classes_name and the second the classes to exclude
    train_dataset = ClassSpecificImageFolderNotAlphabetic(train_dir, all_dropped_classes=classes_name, transform=transform)
    test_dataset = ClassSpecificImageFolderNotAlphabetic(test_dir, all_dropped_classes=classes_name, transform=transform)

    if less_samples:
        evens = list(range(0, len(train_dataset), reduction_factor))
        train_dataset = torch.utils.data.Subset(train_dataset, evens)

    dataset = train_val_dataset(train_dataset, val_split=0.15)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    lr_ratio = 1 / len(train_loader)

    if less_samples:
        class_names = dataset['train'].dataset.dataset.classes
    else:
        class_names = dataset['train'].dataset.classes

    num_class, num_superclass = len(class_names), len(superclasses)

    print(class_names)

    # Network
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features  # input features for the last layers
    model.fc = nn.Linear(num_ftrs, out_features=num_class)
    model.to(device)

    if freeze:
        for name, param in model.named_parameters():
            if param.requires_grad and 'fc' not in name:
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     print(name, param)

    print(summary(model, (3, 32, 32)))

    if not regularization:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps_train = len(train_loader)
    n_total_steps_val = len(val_loader)

    best_acc = 0.0
    associated_sup_acc = 0.0
    platoon = 0
    stop = False

    for epoch in range(num_epochs):

        if stop:
            break

        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print(f"Best acc: {best_acc:.4f}, associate best superclass acc: {associated_sup_acc:.4f}")
        print("-" * 200)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                loader = train_loader
                n_total_steps = n_total_steps_train
            else:
                model.eval()  # Set model to evaluate mode
                loader = val_loader
                n_total_steps = n_total_steps_val

            # vengono set a zero sia per train che per valid
            running_loss = 0.0
            running_corrects = 0

            for i, (images, labels) in enumerate(loader):

                images = images.to(device)
                labels = labels.to(device)

                # forward
                # track history only if train
                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)

                    loss = hierarchical_cc(outputs, labels, np.asarray(coarse_labels), int(num_class/num_superclass), num_superclass,
                                               model, device, hierarchical_loss, regularization, weight_decay=weight_decay)

                    # backward + optimize if training
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()  # * images.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                acc_super = accuracy_superclasses(outputs, labels, np.asarray(coarse_labels), len(superclasses))

                print("Step {}/{}, {} Loss: {:.4f} Acc: {:.4f} Acc Super: {:.4f}".format(i + 1, n_total_steps, phase, epoch_loss, epoch_acc, acc_super))

                if phase == "val" and (i+1) % n_total_steps == 0 and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    associated_sup_acc = acc_super
                    platoon = 0
                    torch.save(model.state_dict(), model_name)
                    print("New best accuracy {:.4f}, superclass accuracy {:.4f}, saving best model".format(best_acc, acc_super))

                if phase == "val" and (i+1) % n_total_steps == 0 and epoch_acc < best_acc:
                    platoon += 1
                    print("{} epochs without improvement".format(platoon))
                    if platoon == early_stopping:
                        print("Network didn't improve after {} epochs, early stopping".format(early_stopping))
                        stop = True

            if phase == "train":
                print("End of training epoch: loss {:.4f}, accuracy {:.4f}".format(epoch_loss, epoch_acc))
                writer.add_scalar("training loss", epoch_loss, epoch)
                writer.add_scalar("training accuracy", epoch_acc, epoch)
                writer.add_scalar("training super accuracy", acc_super, epoch)

            elif phase == "val":
                print("End of validation epoch: loss {:.4f}, accuracy {:.4f}".format(epoch_loss, epoch_acc))
                writer.add_scalar("validation loss", epoch_loss, epoch)
                writer.add_scalar("validation accuracy", epoch_acc, epoch)
                writer.add_scalar("validation super accuracy", acc_super, epoch)

    print("Finished Training")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in class_names], columns=[i for i in class_names])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("output.png")