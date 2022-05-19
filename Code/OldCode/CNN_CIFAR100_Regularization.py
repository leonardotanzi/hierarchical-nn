from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
import torch.nn.functional as F
import os
from utils import ClassSpecificImageFolderNotAlphabetic, imshow, train_val_dataset, sparse2coarse, exclude_classes, \
    get_classes, get_superclasses, accuracy_superclasses
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


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


def classic_cc(predicted, actual):
    # sftmax_out = predicted - predicted.exp().sum(-1).log().unsqueeze(-1) #sftmx = F.log_softmax(predicted, dim=1)
    # actual_onehot = F.one_hot(actual)
    # out1 = actual_onehot * sftmx_out
    # loss = torch.sum(-out1)

    # or
    return torch.sum(-F.one_hot(actual) * (predicted - predicted.exp().sum(-1).log().unsqueeze(-1)))


def hierarchical_cc(predicted, actual, coarse_labels, n_superclass, w_superclasses, w_classes, weight_decay1=None, weight_decay2=None):

    batch = predicted.size(0)

    # compute the loss for fine classes
    loss_fine = F.cross_entropy(predicted, actual, reduction="mean")

    # define an empty vector which contains 20 superclasses prediction for each samples
    predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device="cuda:0")

    for k in range(n_superclass):
        # obtain the indexes of the superclass number k
        indexes = list(np.where(coarse_labels == k))[0]
        # for each index, sum all the probability related to that superclass
        for j in indexes:
            predicted_coarse[:, k] = predicted_coarse[:, k] + predicted[:, j]

    actual_coarse = sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)

    loss_coarse = F.cross_entropy(predicted_coarse, torch.from_numpy(actual_coarse).type(torch.int64).to(device), reduction="mean")

    # I take all the w_classes related to the index. so for example a sample which lable is 2 will get the regularizer
    # in position 2 associated, the same for superclasses

    if weight_decay1 is not None:
        regularizer1 = w_classes[actual]
        regularizer2 = w_superclasses[torch.from_numpy(actual_coarse).type(torch.int64)]

        # return loss_fine + loss_coarse + weight_decay1 * torch.linalg.norm(regularizer1) + weight_decay2 * torch.linalg.norm(regularizer2)
        return loss_fine + weight_decay1 * torch.linalg.norm(regularizer1) + weight_decay2 * torch.linalg.norm(regularizer2)

    return loss_fine + loss_coarse


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dir = "../../../cifar/train//"
    test_dir = "../../../cifar/test//"

    image_size = 32

    num_epochs = 1000
    batch_size = 128
    learning_rate = 0.001
    early_stopping = 200

    model_name = "..//..//cnn_regularization_half.pth"
    hierarchical_loss = True
    weight_decay1 = 0.1
    weight_decay2 = 0.1
    all_superclasses = False
    less_samples = True

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
        evens = list(range(0, len(train_dataset), 2))
        train_dataset = torch.utils.data.Subset(train_dataset, evens)

    dataset = train_val_dataset(train_dataset, val_split=0.15)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}
    if less_samples:
        class_names = dataset['train'].dataset.dataset.classes
    else:
        class_names = dataset['train'].dataset.classes

    num_class, num_superclass = len(class_names), len(superclasses)

    print(class_names)

    # Network
    model = ConvNet(num_class).to(device)

    # H is the number of element in the each regulizing vector
    hidden_dimension = 1024
    # define one (num_superclass, H) vector and (num_class, H) vector, each one contains the values for each
    # regularizing vector in the tree
    w_superclasses = Variable(
        torch.empty(size=(num_superclass, hidden_dimension)).normal_(mean=0, std=1.0).type(torch.FloatTensor),
        requires_grad=True)
    w_classes = Variable(
        torch.empty(size=(num_class, hidden_dimension)).normal_(mean=0, std=1.0).type(torch.FloatTensor),
        requires_grad=True)

    # Optimizer
    if weight_decay1 is not None:

        optimizer = torch.optim.SGD([
            {'params': model.parameters()},
            {'params': [w_superclasses, w_classes]}
        ], lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(reduction="mean")

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
        print("-" * 30)

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

                    if hierarchical_loss:
                        loss = hierarchical_cc(outputs, labels, np.asarray(coarse_labels), len(superclasses),
                                               w_superclasses, w_classes, weight_decay1=None, weight_decay2=None)
                    else:
                        loss = F.cross_entropy(outputs, labels, reduction="mean")

                    # backward + optimize if training
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

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
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in class_names], columns=[i for i in class_names])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("output.png")