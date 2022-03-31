import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import train_val_dataset, hierarchical_cc, get_superclasses, exclude_classes, \
    class_specific_image_folder_not_alphabetic, get_classes, accuracy_superclasses, Identity
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import copy
from torch.optim import lr_scheduler


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    # test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)

    # take as input a list of list with the first element being the classes_name and the second the classes to exclude
    train_dir = "..//..//cifar//train//"
    test_dir = "..//..//cifar//test//"
    # prepare superclasses
    superclasses = get_superclasses()
    classes_name = get_classes()
    # given the list of superclasses, returns the class to exclude and the coarse label
    n_classes = len(classes_name[0])
    n_superclasses = len(superclasses)
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)
    classes_name.append(excluded)
    train_dataset = class_specific_image_folder_not_alphabetic(train_dir, all_dropped_classes=classes_name,
                                                          transform=transform)
    test_dataset = class_specific_image_folder_not_alphabetic(test_dir, all_dropped_classes=classes_name,
                                                         transform=transform)

    batch_size = 128
    n_epochs = 100
    learning_rate = 0.001
    step_size = 40

    hierarchical_loss = False
    regularization = False

    run_scheduler = False
    sp_regularization = True
    weight_decay = 0.01
    less_samples = False
    reduction_factor = 1

    if hierarchical_loss and not regularization:
        model_name = "..//..//Models//New_290322//resnet_hloss_1on{}.pth".format(reduction_factor)
    elif regularization and not hierarchical_loss:
        model_name = "..//..//Models//New_290322//resnet_reg_1on{}.pth".format(reduction_factor)
    elif regularization and hierarchical_loss:
        model_name = "..//..//Models//New_290322//resnet_hloss_reg_1on{}.pth".format(reduction_factor)
    else:
        model_name = "..//..//Models//New_290322//resnet_1on{}.pth".format(reduction_factor)

    model_name = "..//..//Models//New_290322//resnet1on1spfaster.pth".format(reduction_factor)

    writer = SummaryWriter(os.path.join("..//Logs//New_290322//", model_name.split("//")[-1].split(".")[0]))

    # I should apply this just to train not to validation!
    if less_samples:
        evens = list(range(0, len(train_dataset), reduction_factor))
        train_dataset = torch.utils.data.Subset(train_dataset, evens)

    dataset = train_val_dataset(train_dataset, val_split=0.15)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    model = models.resnet18(pretrained=True)
    model_0 = copy.deepcopy(model).to(device)
    model_0.fc = Identity()

    w0 = 0
    if sp_regularization:
        for i, (name, W0) in enumerate(model_0.named_parameters()):
            if 'weight' in name:
                w0 = W0.view(-1) if i == 0 else torch.cat((w0, W0.view(-1)))
        w0 = w0.detach()

    num_ftrs = model.fc.in_features  # input features for the last layers
    model.fc = nn.Linear(num_ftrs, out_features=n_classes)  # we have 2 classes now
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.3)  # every 25 epoch the lr is multiplied by gamma

    n_total_steps_train = len(train_loader)
    n_total_steps_val = len(val_loader)
    platoon = 0
    best_acc = 0.0
    associated_sup_acc = 0.0

    for epoch in range(n_epochs):
        print('-' * 200)
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))

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

            # Iterate over data.
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #loss = F.cross_entropy(outputs, labels)
                    loss = hierarchical_cc(outputs, labels, np.asarray(coarse_labels), int(n_classes / n_superclasses),
                                           n_superclasses, model, w0, device, hierarchical_loss, regularization,
                                           sp_regularization, weight_decay)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) #multiple the loss for the number of the sample in the batch in order to average it
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase] #average the loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            acc_super = accuracy_superclasses(outputs, labels, np.asarray(coarse_labels), len(superclasses))

            print("Step {}/{}, {} Loss: {:.4f} Acc: {:.4f} Acc Super: {:.4f}".format(i + 1, n_total_steps, phase, epoch_loss, epoch_acc, acc_super))

            if (i + 1) % n_total_steps == 0:
                if phase == "val":
                    if epoch_acc.item() > best_acc:
                        best_acc = epoch_acc
                        associated_sup_acc = acc_super
                        platoon = 0
                        torch.save(model.state_dict(), model_name)
                        print("New best accuracy {:.4f}, saving best model".format(best_acc))

                    if epoch_acc.item() < best_acc:
                        platoon += 1
                        print("{} epochs without improvement, best accuracy: {:.4f}".format(platoon, best_acc))

                    print("End of validation epoch: loss {:.4f}, accuracy {:.4f}".format(epoch_loss, epoch_acc))
                    writer.add_scalar("validation loss", epoch_loss, epoch)
                    writer.add_scalar("validation accuracy", epoch_acc, epoch)
                    writer.add_scalar("validation super accuracy", acc_super, epoch)

                elif phase == "train":
                    if run_scheduler:
                        scheduler.step()
                    print("End of training epoch: loss {:.4f}, accuracy {:.4f}".format(epoch_loss, epoch_acc))
                    writer.add_scalar("training loss", epoch_loss, epoch)
                    writer.add_scalar("training accuracy", epoch_acc, epoch)
                    writer.add_scalar("training super accuracy", acc_super, epoch)



