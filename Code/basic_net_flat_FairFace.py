import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torchvision import datasets

from evaluation import accuracy_coarser_classes
from losses import cross_entropy
from dataset import train_val_dataset, ImageFolderNotAlphabetic
from utils import decimal_to_string, seed_everything

import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import timeit


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    batch_size = 128
    n_epochs = 100
    learning_rate = 0.001
    scheduler_step_size = 40
    validation_split = 0.1

    classes = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-99"]

    name = "resnet-fairface-flat"

    run_scheduler = False
    sp_regularization = False
    weight_decay = 0.1
    less_samples = False
    reduction_factor = 1 if less_samples is False else 16
    freeze = False

    # Path
    model_path = "..//..//Models//Mat_version_210622//"
    model_name = os.path.join(model_path,
                                  f"{name}_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    print(f"Model name: {model_name}")

    # Log
    writer = SummaryWriter(os.path.join("..//Logs//Mat_version_210622//", model_name.split("//")[-1].split(".")[0]))

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dir = "..//..//Dataset//FairFace//FairFace_flat"

    # Load the data: train and test sets
    train_dataset = ImageFolderNotAlphabetic(train_dir, classes=classes, transform=transform)

    dataset = train_val_dataset(train_dataset, validation_split, reduction_factor)

    # with open("pkl//fairfaces_dataset.pkl", "wb") as f:
    #     pickle.dump(dataset, f)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4,)
    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    # Check lr_ratio
    lr_ratio = 1 / len(train_loader)
    print(f"LR should be around {lr_ratio:.4f}")

    # Model
    model = models.resnet18(pretrained=True)
    # Freeze layers
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Add last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features=len(classes))
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

            # Iterate over data
            for j, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    outputs = torch.softmax(outputs, dim=1) + 1e-6
                    loss = cross_entropy(outputs, labels, reduction="sum")

                    # Backward + optimize
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"Step {j + 1}/{n_total_steps}, {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if (j + 1) % n_total_steps == 0:
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

                    writer.add_scalars("Training vs. validation loss",
                                       {"Training": epoch_loss_compare, "Validation": epoch_loss}, epoch)
                    writer.add_scalars("Training vs. validation accuracy",
                                       {"Training": epoch_acc_compare, "Validation": epoch_acc}, epoch)


                elif phase == "train":
                    if run_scheduler:
                        scheduler.step()
                    print("End of training epoch.")
                    writer.add_scalar("Training loss", epoch_loss, epoch)
                    writer.add_scalar("Training accuracy", epoch_acc, epoch)

                    epoch_loss_compare = epoch_loss
                    epoch_acc_compare = epoch_acc

        stop = timeit.default_timer()
        print(f"Elapsed time {stop-start:.4f}")

    last_model_name = model_name[:-4] + "_last.pth"
    torch.save(model.state_dict(), last_model_name)

    writer.flush()
    writer.close()
