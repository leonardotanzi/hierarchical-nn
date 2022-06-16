import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import get_superclasses, get_classes, get_medium_labels, sparser2coarser
from evaluation import accuracy_coarser_classes
from losses import hierarchical_cc
from dataset import train_val_dataset, ImageFolderNotAlphabetic
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch.nn.functional as F


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # take as input a list of list with the first element being the classes_name and the second the classes to exclude
    train_dir = "..//..//cifar//train//"
    test_dir = "..//..//cifar//test//"
    # prepare superclasses
    superclasses = get_superclasses()
    classes = get_classes()
    # given the list of superclasses, returns the class to exclude and the coarse label
    n_classes = len(classes)
    n_superclasses = len(superclasses)
    medium_labels = get_medium_labels(superclasses_names=superclasses)

    train_dataset = ImageFolderNotAlphabetic(train_dir, classes=classes, transform=transform)

    batch_size = 128
    n_epochs = 200
    learning_rate = 0.001
    step_size = 40

    hierarchical_loss = False
    regularization = False

    run_scheduler = False
    sp_regularization = False
    weight_decay = 0.1
    less_samples = True
    reduction_factor = 1 if less_samples is False else 16

    if hierarchical_loss and not regularization:
        model_name = "..//..//Models//New_020622//resnet_hloss_1on{}.pth".format(reduction_factor)
    elif regularization and not hierarchical_loss:
        model_name = "..//..//Models//New_020622//resnet_reg_1on{}.pth".format(reduction_factor)
    elif regularization and hierarchical_loss:
        model_name = "..//..//Models//New_020622//resnet_hloss_reg_1on{}.pth".format(reduction_factor)
    else:
        model_name = "..//..//Models//New_020622//resnet_1on{}.pth".format(reduction_factor)

    writer = SummaryWriter(os.path.join("..//Logs//New_020622//", model_name.split("//")[-1].split(".")[0]))

    dataset = train_val_dataset(train_dataset, 0.15, reduction_factor)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    lr_ratio = 1 / len(train_loader)

    # get some random training images and plot
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # print(' '.join("%s" % classes[0][labels[j]] for j in range(10)))
    # imshow(torchvision.utils.make_grid(images))

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # #for sp regularization
    vec = []
    for i, (name, W) in enumerate(model.named_parameters()):
        if 'weight' in name and 'fc' not in name:
            vec.append(W.view(-1))
    w0 = torch.cat(vec).detach().to(device)

    num_ftrs = model.fc.in_features  # input features for the last layers
    model.fc = nn.Linear(num_ftrs, out_features=n_superclasses)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if regularization else torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if run_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                        gamma=0.3)  # every n=step_size epoch the lr is multiplied by gamma

    n_total_steps_train = len(train_loader)
    n_total_steps_val = len(val_loader)
    platoon = 0
    best_acc = 0.0
    associated_sup_acc = 0.0

    for epoch in range(n_epochs):
        print('-' * 200)
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                loader = train_loader
                n_total_steps = n_total_steps_train
            else:
                model.eval()  # Set model to evaluate mode
                loader = val_loader
                n_total_steps = n_total_steps_val

            running_loss = 0.0
            running_corrects = 0
            running_corrects_super = 0
            running_loss_fine = 0.0
            running_loss_medium = 0.0
            running_medium_penalty = 0.0
            running_fine_penalty = 0.0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = torch.from_numpy(sparser2coarser(labels.numpy(), np.asarray(medium_labels)))
                labels = labels.type(torch.int64)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = F.cross_entropy(outputs, labels, reduction="sum")

                    loss_dict = {"loss_fine": 0.0, "loss_medium": 0.0, "fine_penalty": 0.0, "medium_penalty": 0.0}

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                running_loss_fine += loss_dict["loss_fine"]
                running_loss_medium += loss_dict["loss_medium"]
                running_medium_penalty += loss_dict["medium_penalty"]
                running_fine_penalty += loss_dict["fine_penalty"]

            epoch_loss = running_loss / dataset_sizes[phase]  # average the loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss_fine = running_loss_fine / dataset_sizes[phase]
            epoch_loss_medium = running_loss_medium / dataset_sizes[phase]
            epoch_medium_penalty = running_medium_penalty / dataset_sizes[phase]
            epoch_fine_penalty = running_fine_penalty / dataset_sizes[phase]

            print("Step {}/{}, {} Loss: {:.4f} Acc: {:.4f}".format(i + 1, n_total_steps, phase, epoch_loss, epoch_acc))

            if (i + 1) % n_total_steps == 0:
                if phase == "val":
                    if epoch_acc.item() > best_acc:
                        best_acc = epoch_acc
                        platoon = 0
                        torch.save(model.state_dict(), model_name)
                        print("New best accuracy {:.4f}, saving best model".format(best_acc))

                    if epoch_acc.item() < best_acc:
                        platoon += 1
                        print("{} epochs without improvement, best accuracy: {:.4f}".format(platoon, best_acc))

                    print("End of validation epoch.")
                    writer.add_scalar("validation loss", epoch_loss, epoch)
                    writer.add_scalar("validation accuracy", epoch_acc, epoch)
                    writer.add_scalars('training vs. validation loss',
                                       {'training': epoch_loss_t, 'validation': epoch_loss}, epoch)
                    writer.add_scalars('training vs. validation accuracy',
                                       {'training': epoch_acc_t, 'validation': epoch_acc}, epoch)
                    writer.add_scalars('4 losses validation', {'Loss': epoch_loss,
                                                               'Fine Loss': epoch_loss_fine,
                                                               'Medium Loss': epoch_loss_medium,
                                                               'Fine Penalty': epoch_fine_penalty,
                                                               'Medium Penalty': epoch_medium_penalty,
                                                               'Loss with WD applied': epoch_loss * weight_decay*(epoch_fine_penalty + epoch_medium_penalty)}, epoch)
                elif phase == "train":
                    if run_scheduler:
                        scheduler.step()
                    print("End of training epoch.")
                    writer.add_scalar("training loss", epoch_loss, epoch)
                    writer.add_scalar("training accuracy", epoch_acc, epoch)
                    writer.add_scalars('4 losses training', {'Loss': epoch_loss,
                                                             'Fine Loss': epoch_loss_fine,
                                                             'Medium Loss': epoch_loss_medium,
                                                             'Fine Penalty': epoch_fine_penalty,
                                                             'Medium Penalty': epoch_medium_penalty,
                                                             'Loss with WD applied':
                                                                   epoch_loss * weight_decay * (epoch_fine_penalty + epoch_medium_penalty)}, epoch)
                    epoch_loss_t = epoch_loss
                    epoch_acc_t = epoch_acc
                    
    writer.flush()
    writer.close()
