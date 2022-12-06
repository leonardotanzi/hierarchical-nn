import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomRotation, \
    GaussianBlur, RandomErasing, RandomPerspective, CenterCrop, RandomVerticalFlip
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

import cv2
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
from torchsampler import ImbalancedDatasetSampler

# Use SMOTE algo
# Apply some preprocessing to images
# Try strong regularization
# change image size to 224

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    architecture = "densenet"

    n_epochs = 60
    image_size = 224
    validation_split = 0.15
    batch_size = 256
    freeze = False
    run_scheduler = True
    load_model = False
    learning_rate = 1e-4
    weight_decay = 0.5
    scheduler_step_size = 15
    classes = [0, 1]
    n_output = len(classes)

    model_name = f"..//..//..//methinks//Models//skull_{architecture}_lr{str(learning_rate)}_wd{str(weight_decay)}_oversampling_aug"

    writer = SummaryWriter(os.path.join("..//..//..//methinks//Logs", model_name))

    basic_transform = Compose([ToTensor(),
                               Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               Resize((image_size, image_size))])

    augmentation_transform = Compose([ToTensor(),
                                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    Resize(image_size),
                                    CenterCrop(200),
                                    RandomHorizontalFlip(p=0.5),
                                    RandomVerticalFlip(p=0.5),
                                    RandomRotation(degrees=30),
                                    # GaussianBlur(kernel_size=(51, 91), sigma=3),
                                    # RandomErasing(p=0.3),
                                    RandomPerspective(p=0.3, distortion_scale=0.3)])

    train_dir = "..//..//..//methinks//cleaned_brain//train"
    test_dir = "..//..//..//methinks//cleaned_brain//test"

    train_dataset = ImageFolder(train_dir, augmentation_transform)
    test_dataset = ImageFolder(test_dir, basic_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

    if architecture == "inception":
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
        # Freeze layers
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        # model.fc = nn.Sequential(nn.Linear(num_ftrs, out_features=1024), nn.LeakyReLU(), nn.Linear(1024, n_output))
        model.fc = nn.Linear(num_ftrs, n_output)

    elif architecture == "resnet":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        # model.fc = nn.Sequential(nn.Linear(num_ftrs, out_features=1024), nn.LeakyReLU(), nn.Linear(1024, n_output))
        model.fc = nn.Linear(num_ftrs, n_output)

    elif architecture == "densenet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        num_ftrs = model.classifier.in_features
        # model.fc = nn.Sequential(nn.Linear(num_ftrs, out_features=1024), nn.LeakyReLU(), nn.Linear(1024, n_output))
        model.classifier = nn.Linear(num_ftrs, n_output)

    if load_model:
        model.load_state_dict(torch.load("skull_pretr21_best.pth"))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Scheduler
    if run_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.3)
        # every n=step_size epoch the lr is multiplied by gamma

    n_total_steps_train = len(train_loader)
    n_total_steps_val = len(test_loader)
    platoon = 0
    best_acc = 0.0

    for epoch in range(n_epochs):
        print("-" * 200)
        print(f"Epoch {epoch + 1}/{n_epochs}")
        CM = 0

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
                loader = train_loader
                n_total_steps = n_total_steps_train
            else:
                model.eval()
                loader = test_loader
                n_total_steps = n_total_steps_val

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for j, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                print(f"Step {j} / {len(loader)}")

                # Forward
                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = F.cross_entropy(outputs, labels, weight=torch.FloatTensor([0.5, 0.5]).to(device))

                    # Backward + optimize
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    if phase == "test":
                        CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])
                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"Step {j + 1}/{n_total_steps}, {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if (j + 1) % n_total_steps == 0:
                if phase == "test":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        platoon = 0
                        best_model_name = model_name + f"{epoch}_best.pth"
                        torch.save(model.state_dict(), best_model_name)
                        print(f"New best accuracy {best_acc:.4f}, saving best model")

                    if epoch_acc < best_acc:
                        platoon += 1
                        print(f"{platoon} epochs without improvement, best accuracy: {best_acc:.4f}")

                    print("End of validation epoch.")
                    writer.add_scalar("Validation loss", epoch_loss, epoch)
                    writer.add_scalar("Validation accuracy", epoch_acc, epoch)

                    writer.add_scalars("Training vs. validation loss",
                                       {"Training": epoch_loss_compare, "Validation": epoch_loss}, epoch)
                    writer.add_scalars("Training vs. validation accuracy",
                                       {"Training": epoch_acc_compare, "Validation": epoch_acc}, epoch)

                    tn = CM[0][0]
                    tp = CM[1][1]
                    fp = CM[0][1]
                    fn = CM[1][0]
                    acc = np.sum(np.diag(CM) / np.sum(CM))
                    sensitivity = tp / (tp + fn)
                    precision = tp / (tp + fp)

                    print('\nTestset Accuracy (mean): %f %%' % (100 * acc))
                    print()
                    print('Confusion Matrix : ')
                    print(CM)
                    print('- Sensitivity : ', (tp / (tp + fn)) * 100)
                    print('- Specificity : ', (tn / (tn + fp)) * 100)

                    torch.save(model.state_dict(), model_name + f"{epoch}.pth")

                elif phase == "train":
                    if run_scheduler:
                        scheduler.step()

                    print("End of training epoch.")
                    writer.add_scalar("Training loss", epoch_loss, epoch)
                    writer.add_scalar("Training accuracy", epoch_acc, epoch)

                    epoch_loss_compare = epoch_loss
                    epoch_acc_compare = epoch_acc

    last_model_name = model_name + "_last.pth"

    # test
    CM = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # file_name
            preds = torch.argmax(outputs.data, 1)
            CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])

        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        acc = np.sum(np.diag(CM) / np.sum(CM))
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)

        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM)
        print('- Sensitivity : ', (tp / (tp + fn)) * 100)
        print('- Specificity : ', (tn / (tn + fp)) * 100)

    writer.flush()
    writer.close()
