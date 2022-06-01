import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import train_val_dataset, hierarchical_cc, get_superclasses, exclude_classes, \
    ClassSpecificImageFolderNotAlphabetic, get_classes, accuracy_superclasses, return_tree_CIFAR, \
    imshow, select_n_random, ConvNet, sparse2coarse, decimal_to_string
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch.nn.functional as F
import timeit
import random


if __name__ == "__main__":

    # H- Parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    n_epochs = 100
    learning_rate = 0.001
    scheduler_step_size = 40
    validation_split = 0.15

    hierarchical_loss = True
    regularization = True
    name = "invertedsup"

    run_scheduler = False
    sp_regularization = False
    weight_decay = 0.1
    less_samples = True
    reduction_factor = 1 if less_samples is False else 16

    # Classes and superclasses
    classes = get_classes()
    random.seed(0)
    random.shuffle(classes[0])
    superclasses = get_superclasses()
    n_classes = len(classes[0])
    n_superclasses = len(superclasses)
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)
    classes.append(excluded)

    # Path
    model_path = "..//..//Models//Final_100522//"
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
    writer = SummaryWriter(os.path.join("..//Logs//Final_100522//", model_name.split("//")[-1].split(".")[0]))

    # Dataset
    train_dir = "..//..//cifar//train//"
    test_dir = "..//..//cifar//test//"
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = ClassSpecificImageFolderNotAlphabetic(train_dir, all_dropped_classes=classes, transform=transform)
    dataset = train_val_dataset(train_dataset, validation_split, reduction_factor)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4,
                            pin_memory=True)
    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    # Check lr_ratio
    lr_ratio = 1 / len(train_loader)
    print(f"LR should be around {lr_ratio:.4f}")

    # Plot
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # print(" ".join("%s" % classes[0][labels[j]] for j in range(10)))
    # imshow(torchvision.utils.make_grid(images))

    # Model
    model = models.resnet18(pretrained=True)
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    # SP Reg
    if sp_regularization:
        vec = []
        for i, (name, W) in enumerate(model.named_parameters()):
            if "weight" in name and "fc" not in name:
                vec.append(W.view(-1))
        w0 = torch.cat(vec).detach().to(device)
    else:
        w0 = None

    # Add last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features=n_classes)
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
    associated_sup_acc = 0.0

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
            running_corrects_super = 0
            running_loss_fine = 0.0
            running_loss_coarse = 0.0
            running_coarse_penalty = 0.0
            running_fine_penalty = 0.0

            # Iterate over data
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss, loss_dict = hierarchical_cc(outputs, labels, np.asarray(coarse_labels),
                                                      return_tree_CIFAR(), int(n_classes / n_superclasses),
                                                      n_superclasses, model, w0, device, hierarchical_loss,
                                                      regularization, sp_regularization, weight_decay)

                    # Backward + optimize
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
                running_corrects_super += accuracy_superclasses(outputs, labels, np.asarray(coarse_labels),
                                                                len(superclasses), device)
                running_loss_fine += loss_dict["loss_fine"]
                running_loss_coarse += loss_dict["loss_coarse"]
                running_coarse_penalty += loss_dict["coarse_penalty"]
                running_fine_penalty += loss_dict["fine_penalty"]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_acc_super = running_corrects_super / dataset_sizes[phase]
            epoch_loss_fine = running_loss_fine / dataset_sizes[phase]
            epoch_loss_coarse = running_loss_coarse / dataset_sizes[phase]
            epoch_coarse_penalty = running_coarse_penalty / dataset_sizes[phase]
            epoch_fine_penalty = running_fine_penalty / dataset_sizes[phase]

            print(f"Step {i + 1}/{n_total_steps}, {phase} Loss: {epoch_loss:.4f},"
                  f" Acc: {epoch_acc:.4f}, Acc Super: {epoch_acc_super:.4f}")

            if (i + 1) % n_total_steps == 0:
                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        associated_sup_acc = epoch_acc_super
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
                    writer.add_scalar("Validation super accuracy", epoch_acc_super, epoch)
                    writer.add_scalars("Training vs. validation loss",
                                       {"Training": epoch_loss_compare, "Validation": epoch_loss}, epoch)
                    writer.add_scalars("Training vs. validation accuracy",
                                       {"Training": epoch_acc_compare, "Validation": epoch_acc}, epoch)
                    writer.add_scalars("Losses and penalties validation", {"Loss": epoch_loss,
                                                                           "Fine Loss": epoch_loss_fine,
                                                                           "Coarse Loss": epoch_loss_coarse,
                                                                           "Fine Penalty": epoch_fine_penalty,
                                                                           "Coarse Penalty": epoch_coarse_penalty},
                                       epoch)

                elif phase == "train":
                    if run_scheduler:
                        scheduler.step()
                    print("End of training epoch.")
                    writer.add_scalar("Training loss", epoch_loss, epoch)
                    writer.add_scalar("Training accuracy", epoch_acc, epoch)
                    writer.add_scalar("Training super accuracy", epoch_acc_super, epoch)
                    writer.add_scalars("Losses and penalties training", {"Loss": epoch_loss,
                                                                         "Fine Loss": epoch_loss_fine,
                                                                         "Coarse Loss": epoch_loss_coarse,
                                                                         "Fine Penalty": epoch_fine_penalty,
                                                                         "Coarse Penalty": epoch_coarse_penalty},
                                       epoch)
                    epoch_loss_compare = epoch_loss
                    epoch_acc_compare = epoch_acc

        stop = timeit.default_timer()
        print(f"Elapsed time {stop-start:.4f}")

    last_model_name = model_name[:-4] + "_last.pth"
    torch.save(model.state_dict(), last_model_name)

    writer.flush()
    writer.close()
