import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import train_val_dataset, hierarchical_cc, get_superclasses, exclude_classes, \
    class_specific_image_folder_not_alphabetic, get_classes, accuracy_superclasses, return_tree_CIFAR, \
    imshow, select_n_random, hierarchical_cc_singlelosses, hierarchical_cc_singlelosses_regularization
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from collections import Counter
import torchvision


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # train_dataset_cifar = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    # test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)

    # take as input a list of list with the first element being the classes_name and the second the classes to exclude
    train_dir = "..//..//cifar//train//"
    test_dir = "..//..//cifar//test//"
    # prepare superclasses
    superclasses = get_superclasses()
    # superclasses = ["aquatic mammals", "fish"]
    classes = get_classes()
    # given the list of superclasses, returns the class to exclude and the coarse label
    n_classes = len(classes[0])
    n_superclasses = len(superclasses)
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)
    classes.append(excluded)
    train_dataset = class_specific_image_folder_not_alphabetic(train_dir, all_dropped_classes=classes,
                                                          transform=transform)

    batch_size = 128
    n_epochs = 100
    learning_rate = 0.001
    step_size = 40

    hierarchical_loss = True
    regularization = False

    run_scheduler = False
    sp_regularization = False
    weight_decay = 1
    less_samples = False
    reduction_factor = 1 if less_samples is False else 2

    if hierarchical_loss and not regularization:
        model_name = "..//..//Models//New_110422//resnet_hloss_1on{}.pth".format(reduction_factor)
    elif regularization and not hierarchical_loss:
        model_name = "..//..//Models//New_110422//resnet_reg_1on{}.pth".format(reduction_factor)
    elif regularization and hierarchical_loss:
        model_name = "..//..//Models//New_110422//resnet_hloss_reg_1on{}.pth".format(reduction_factor)
    else:
        model_name = "..//..//Models//New_110422//resnet_1on{}.pth".format(reduction_factor)

    model_name = "..//..//Models//New_110422//resnet_new_1on{}.pth".format(reduction_factor)

    writer = SummaryWriter(os.path.join("..//Logs//New_110422//", model_name.split("//")[-1].split(".")[0]))

    dataset = train_val_dataset(train_dataset, 0.15, reduction_factor)

    # if less_samples:
    #     evens = list(range(0, len(dataset["train"]), reduction_factor))
    #     dataset["train"] = torch.utils.data.Subset(dataset["train"], evens)
    #     train_classes = [dataset["train"].dataset.dataset.targets[i] for i in dataset["train"].indices]
    #     print(Counter(train_classes))
    #     val_classes = [dataset["val"].dataset.targets[i] for i in dataset["val"].indices]
    #     print(Counter(val_classes))

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    lr_ratio = 1 / len(train_loader)

    # get some random training images and plot
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # print(' '.join("%s" % classes[0][labels[j]] for j in range(10)))
    # imshow(torchvision.utils.make_grid(images))

    # # Extract a random subset of data
    # images, labels = select_n_random(train_dataset_cifar.data, train_dataset_cifar.targets)
    # # get the class labels for each image
    # class_labels = [classes[label] for label in labels]
    # # log embeddings
    # features = images.view(-1, 32 * 32)
    # writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    model = models.resnet18(pretrained=True)

    #for sp regularization
    vec = []
    for i, (name, W) in enumerate(model.named_parameters()):
        if 'weight' in name and 'fc' not in name:
            vec.append(W.view(-1))
    w0 = torch.cat(vec).detach().to(device)

    num_ftrs = model.fc.in_features  # input features for the last layers
    model.fc = nn.Linear(num_ftrs, out_features=n_classes)  # we have 2 classes now
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) if regularization else torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if run_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.3)  # every n=step_size epoch the lr is multiplied by gamma

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
            running_loss_fine = 0.0
            running_loss_coarse = 0.0
            running_loss_sp = 0.0
            running_loss_reg = 0.0
            running_corrects = 0
            running_corrects_super = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss, loss_fine, loss_coarse, loss_reg, loss_sp = hierarchical_cc_singlelosses(outputs, labels, np.asarray(coarse_labels),
                                            n_superclasses, model, w0, device, weight_decay)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) #multiple the loss for the number of the sample in the batch in order to average it
                running_loss_fine += loss_fine.item() * inputs.size(0)
                running_loss_coarse += loss_coarse.item() * inputs.size(0)
                running_loss_sp += loss_sp.item() * inputs.size(0)
                running_loss_reg += loss_reg.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                running_corrects_super += accuracy_superclasses(outputs, labels, np.asarray(coarse_labels), len(superclasses))

            epoch_loss = running_loss / dataset_sizes[phase] #average the loss
            epoch_loss_fine = running_loss_fine / dataset_sizes[phase]  # average the loss
            epoch_loss_coarse = running_loss_coarse / dataset_sizes[phase]  # average the loss
            epoch_loss_sp = running_loss_sp / dataset_sizes[phase]  # average the loss
            epoch_loss_reg = running_loss_reg / dataset_sizes[phase]  # average the loss

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            acc_super = running_corrects_super / dataset_sizes[phase]

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

                    print("End of validation epoch.")
                    writer.add_scalar("validation loss", epoch_loss, epoch)
                    writer.add_scalar("validation accuracy", epoch_acc, epoch)
                    writer.add_scalar("validation super accuracy", acc_super, epoch)
                    writer.add_scalars('training vs. validation loss', {'training': epoch_loss_t, 'validation': epoch_loss}, epoch)
                    writer.add_scalars('4 losses validation', {'Loss': epoch_loss,
                                                               'Fine Loss': epoch_loss_fine,
                                                               'Coarse Loss': epoch_loss_coarse,
                                                               'Loss SP': epoch_loss_sp,
                                                               'Loss Regularization': epoch_loss_reg}, epoch)

                elif phase == "train":
                    if run_scheduler:
                        scheduler.step()
                    print("End of training epoch.")
                    writer.add_scalar("training loss", epoch_loss, epoch)
                    writer.add_scalar("training accuracy", epoch_acc, epoch)
                    writer.add_scalar("training super accuracy", acc_super, epoch)
                    epoch_loss_t = epoch_loss
                    epoch_acc_t = epoch_acc
                    acc_super_t = acc_super
                    writer.add_scalars('training vs. validation accuracy', {'training': epoch_acc_t, 'validation': epoch_acc}, epoch)
                    writer.add_scalars('4 losses training', {'Loss': epoch_loss,
                                                               'Fine Loss': epoch_loss_fine,
                                                               'Coarse Loss': epoch_loss_coarse,
                                                               'Loss SP': epoch_loss_sp,
                                                               'Loss Regularization': epoch_loss_reg}, epoch)

    writer.flush()
    writer.close()