import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
import torch.nn.functional as F
from torchsummary import summary
from torch.optim import SGD
from torchvision import datasets, models
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import os
from utils import ClassSpecificImageFolderNotAlphabetic, imshow, train_val_dataset, sparse2coarse, exclude_classes, \
    get_classes, get_superclasses
import sys
from torch.utils.tensorboard import SummaryWriter


def hierarchical_cc(predicted, actual, coarse_labels, n_superclass):

    batch = predicted.size(0)

    # compute the loss for fine classes
    loss_fine = F.cross_entropy(predicted, actual)

    # define an empty vector which contains 20 superclasses prediction for each samples
    predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device="cuda:0")

    for k in range(n_superclass):
        # obtain the indexes of the superclass number k
        indexes = list(np.where(coarse_labels == k))[0]
        # for each index, sum all the probability related to that superclass
        for j in indexes:
            predicted_coarse[:, k] = predicted_coarse[:, k] + predicted[:, j]

    actual_coarse = sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)

    loss_coarse = F.cross_entropy(predicted_coarse, torch.from_numpy(actual_coarse).type(torch.int64).to(device))

    return loss_fine + loss_coarse


def classic_cc(predicted, actual):
    # sftmax_out = predicted - predicted.exp().sum(-1).log().unsqueeze(-1) #sftmx = F.log_softmax(predicted, dim=1)
    # actual_onehot = F.one_hot(actual)
    # out1 = actual_onehot * sftmx_out
    # loss = torch.sum(-out1)

    # or
    return torch.sum(-F.one_hot(actual) * (predicted - predicted.exp().sum(-1).log().unsqueeze(-1)))


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same")
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dir = "../../../cifar/train//"
    test_dir = "../../../cifar/test//"

    model_name = "..//..//cnn_hierarchical_allclasses.pth"

    writer = SummaryWriter(os.path.join("..//Logs//", model_name.split("//")[-1].split(".")[0]))
    classes_name = get_classes()

    superclasses = ["flowers", "fruit and vegetables", "trees"]
    # superclasses = get_superclasses()

    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)

    classes_name.append(excluded)

    train_dataset = ClassSpecificImageFolderNotAlphabetic(train_dir, all_dropped_classes=classes_name, transform=transform)
    test_dataset = ClassSpecificImageFolderNotAlphabetic(test_dir, all_dropped_classes=classes_name, transform=transform)

    num_epochs = 1000
    batch_size = 128
    learning_rate = 0.001
    image_size = 32

    convnet = True

    dataset = train_val_dataset(train_dataset, val_split=0.15)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}
    class_names = dataset['train'].dataset.classes
    print(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get some random training images and plot
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

    if convnet:
        model = ConvNet(num_classes=len(class_names))

    else:
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features  # input features for the last layers
        model.fc = nn.Linear(num_ftrs, out_features=len(class_names))  # we have 2 classes now

    model.to(device)

    print(summary(model, (3, 32, 32)))
    writer.add_graph(model, images.to(device))

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = SGD(model.parameters(), lr=learning_rate)
    # step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # every 7 epoch the lr is multiplied by this value
    n_total_steps_train = len(train_loader)
    n_total_steps_val = len(val_loader)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs - 1))
        print(f"Best acc: {best_acc:.4f}")
        print("-" * 10)

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
                    loss = hierarchical_cc(outputs, labels, np.asarray(coarse_labels), len(superclasses))
                    # loss = classic_cc(outputs, labels)
                    # backward + optimize if training
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if phase == "train":
                        if (i+1) % n_total_steps == 0:
                            writer.add_scalar("training loss", running_loss / n_total_steps, epoch * n_total_steps + 1)
                            writer.add_scalar("training accuracy", running_corrects / n_total_steps, epoch * n_total_steps + 1)
                    elif phase == "val":
                        if (i+1) % n_total_steps == 0:
                            writer.add_scalar("validation loss", running_loss / n_total_steps, epoch * n_total_steps + 1)
                            writer.add_scalar("validation accuracy", running_corrects / n_total_steps, epoch * n_total_steps + 1)
                # if phase == "train":
                #     step_lr_scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print("Step {}/{}, {} Loss: {:.4f} Acc: {:.4f}".format(i + 1, n_total_steps, phase, epoch_loss, epoch_acc))

                if phase == "val" and (i+1) % n_total_steps == 0 and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), model_name)
                    print("New best accuracy {}, saving best model".format(best_acc))

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