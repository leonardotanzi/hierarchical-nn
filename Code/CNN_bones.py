import torch
import torchvision
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torchsummary import summary
from torch.optim import SGD
from torchvision import datasets, models
import os
import os
import copy
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def hierarchical_cross_entropy(predicted, actual):
    batch = predicted.size(0)
    length = predicted.size(1)
    loss_fine = F.cross_entropy(predicted, actual)

    if length == 7:
        predicted_medium = torch.empty(batch, 3, dtype=torch.float32, device="cuda:0")
        actual_medium = torch.empty(batch, dtype=torch.int64, device="cuda:0")

        for i, sample in enumerate(predicted):
            predicted_medium[i] = torch.tensor([torch.sum(sample[0:3]), torch.sum(sample[3:6]), sample[6]], dtype=torch.float32)

        for j, sample in enumerate(actual):
            if sample.item() == 0 or sample.item() == 1 or sample.item() == 2:
                actual_medium[j] = torch.tensor(0, dtype=torch.int64)
            elif sample.item() == 3 or sample.item() == 4 or sample.item() == 5:
                actual_medium[j] = torch.tensor(1, dtype=torch.int64)
            else:
                actual_medium[j] = torch.tensor(2, dtype=torch.int64)

        loss_medium = F.cross_entropy(predicted_medium, actual_medium)

        predicted_coarse = torch.empty(batch, 2, dtype=torch.float32, device="cuda:0")
        actual_coarse = torch.empty(batch, dtype=torch.int64, device="cuda:0")
        for i, sample in enumerate(predicted_medium):
            predicted_coarse[i] = torch.tensor([torch.sum(sample[0:2]), sample[2]], dtype=torch.float32)

        for j, sample in enumerate(actual_medium):
            if sample.item() == 0 or sample.item() == 1:
                actual_coarse[j] = torch.tensor(0, dtype=torch.int64)
            else:
                actual_coarse[j] = torch.tensor(1, dtype=torch.int64)

        loss_coarse = F.cross_entropy(predicted_coarse, actual_coarse)

        final_loss = loss_fine + loss_medium + loss_coarse

    elif length == 3:
        predicted_coarse = torch.empty(batch, 2, dtype=torch.float32, device="cuda:0")
        actual_coarse = torch.empty(batch, dtype=torch.int64, device="cuda:0")
        for i, sample in enumerate(predicted):
            predicted_coarse[i] = torch.tensor([torch.sum(sample[0:2]), sample[2]], dtype=torch.float32)

        for j, sample in enumerate(actual):
            if sample.item() == 0 or sample.item() == 1:
                actual_coarse[j] = torch.tensor(0, dtype=torch.int64)
            else:
                actual_coarse[j] = torch.tensor(1, dtype=torch.int64)

        loss_coarse = F.cross_entropy(predicted_coarse, actual_coarse)

        final_loss = loss_fine + loss_coarse

    return final_loss


class ClassSpecificImageFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root,
            dropped_classes=[],
            transform=None,
            target_transform=None,
            loader=datasets.folder.default_loader,
            is_valid_file=None,
    ):
        self.dropped_classes = dropped_classes
        super(ClassSpecificImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       is_valid_file=is_valid_file)
        self.imgs = self.samples

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_val_dataset(dataset, val_split=0.15):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, shuffle=True)
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same")
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same")
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
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
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    data_dir = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Train\\'

    num_epochs = 1
    batch_size = 32
    learning_rate = 0.001
    image_size = 224

    convnet = False

    transform = Compose([ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), Resize(size=(image_size, image_size))])
    images_dataset = ClassSpecificImageFolder(data_dir, dropped_classes=["A", "B", "Broken"], transform=transform)
    # images_dataset = ImageFolder(data_dir, transform=transform)
    dataset = train_val_dataset(images_dataset, val_split=0.15)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}
    class_names = dataset['train'].dataset.classes
    print(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get some random training images and plot
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))

    if convnet:
        model = ConvNet(num_classes=len(class_names))

    else:
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features  # input features for the last layers
        model.fc = nn.Linear(num_ftrs, out_features=len(class_names))  # we have 2 classes now

    model.to(device)

    print(summary(model, (3, 224, 224)))

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    # step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # every 7 epoch the lr is multiplied by this value
    n_total_steps = len(train_loader)

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
            else:
                model.eval()  # Set model to evaluate mode
                loader = val_loader

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
                    loss = hierarchical_cross_entropy(outputs, labels)

                    # backward + optimize if training
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # if phase == "train":
                #     step_lr_scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print("Step {}/{}, {} Loss: {:.4f} Acc: {:.4f}".format(i + 1, n_total_steps, phase, epoch_loss, epoch_acc))

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc

    print("Finished Training")
    PATH = "cnn_hierarchical.pth"
    torch.save(model.state_dict(), PATH)

    test_dir = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'
    test_dataset = ClassSpecificImageFolder(test_dir, dropped_classes=["A", "B", "Broken", "TestDoctors"], transform=transform)
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
    plt.savefig('output.png')