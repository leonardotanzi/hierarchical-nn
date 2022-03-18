import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
import torch.nn.functional as F
from utils import ClassSpecificImageFolderNotAlphabetic, train_val_dataset, sparse2coarse, exclude_classes, \
    get_classes, get_superclasses, accuracy_superclasses, hierarchical_cc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import os


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])  #, Resize(size=(224,224))])

    train_dir = "..//..//cifar//train//"
    test_dir = "..//..//cifar//test//"

    num_epochs = 250
    batch_size = 128
    learning_rate = 0.001
    early_stopping = 200

    hierarchical_loss = True
    regularization = True

    weight_decay1 = 0.1
    weight_decay2 = 0.1
    less_samples = True
    reduction_factor = 2

    if hierarchical_loss and not regularization:
        model_name = "..//..//resnet_hloss_1on{}.pth".format(reduction_factor)
    elif regularization and not hierarchical_loss:
        model_name = "..//..//resnet_reg_1on{}.pth".format(reduction_factor)
    elif regularization and hierarchical_loss:
        model_name = "..//..//resnet_hloss_reg_1on{}.pth".format(reduction_factor)
    else:
        model_name = "..//..//resnet_1on{}.pth".format(reduction_factor)

    writer = SummaryWriter(os.path.join("..//Logs//", model_name.split("//")[-1].split(".")[0]))

    classes = get_classes()

    superclasses = get_superclasses()

    # given the list of superclasses, returns the class to exclude and the coarse label
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)

    classes.append(excluded)

    # take as input a list of list with the first element being the classes_name and the second the classes to exclude
    train_dataset = ClassSpecificImageFolderNotAlphabetic(train_dir, all_dropped_classes=classes, transform=transform)
    test_dataset = ClassSpecificImageFolderNotAlphabetic(test_dir, all_dropped_classes=classes, transform=transform)

    if less_samples:
        evens = list(range(0, len(train_dataset), reduction_factor))
        train_dataset = torch.utils.data.Subset(train_dataset, evens)

    dataset = train_val_dataset(train_dataset, val_split=0.15)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    num_class, num_superclass = len(classes[0]), len(superclasses)

    print(dataset_sizes)
    print(classes[0])

    # Import resnet and change last layer
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features  # input features for the last layers
    model.fc = nn.Linear(num_ftrs, out_features=num_class)
    model.to(device)

    # print model summary
    print(summary(model, (3, 32, 32)))

    # add model structure to tensorboard
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    writer.add_graph(model, images.to(device))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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

                    loss = hierarchical_cc(outputs, labels, np.asarray(coarse_labels), int(num_class/num_superclass),
                                           num_superclass, model, device, hierarchical_loss, regularization,
                                           weight_decay1=weight_decay1, weight_decay2=weight_decay2)

                    # backward + optimize if training
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # # tensorboard
                    # if phase == "train":
                    #     if (i+1) % n_total_steps == 0:
                    #         writer.add_scalar("training loss", running_loss / n_total_steps, epoch * n_total_steps + 1)
                    #         writer.add_scalar("training accuracy", running_corrects / n_total_steps, epoch * n_total_steps + 1)
                    # elif phase == "val":
                    #     if (i+1) % n_total_steps == 0:
                    #         writer.add_scalar("validation loss", running_loss / n_total_steps, epoch * n_total_steps + 1)
                    #         writer.add_scalar("validation accuracy", running_corrects / n_total_steps, epoch * n_total_steps + 1)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("output.png")