import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
import torch.nn.functional as F
from utils import ClassSpecificImageFolderNotAlphabetic, imshow, train_val_dataset, sparse2coarse, exclude_classes, \
    get_classes, get_superclasses, accuracy_superclasses, hierarchical_cc, return_tree_CIFAR, decimal_to_string
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), Resize(size=(224, 224))])

    train_dir = "..//..//..//cifar//train//"
    test_dir = "..//..//..//cifar//test//"

    image_size = 32

    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    early_stopping = 200

    hierarchical_loss = False
    regularization = False
    run_scheduler = False
    sp_regularization = False
    w0 = 0.0
    weight_decay = 0.1

    less_samples = False
    reduction_factor = 32

    if hierarchical_loss and not regularization:
        model_name = "..//..//..//Models//Final_100522//vit_hloss_lr{}_wd{}_1on{}.pth".format(
            decimal_to_string(learning_rate), decimal_to_string(weight_decay), reduction_factor)
    elif regularization and not hierarchical_loss:
        model_name = "..//..//..//Models//Final_100522//vit_reg_lr{}_wd{}_1on{}.pth".format(
            decimal_to_string(learning_rate), decimal_to_string(weight_decay), reduction_factor)
    elif regularization and hierarchical_loss:
        model_name = "..//..//..//Models//Final_100522//vit_hloss_reg_lr{}_wd{}_1on{}.pth".format(
            decimal_to_string(learning_rate), decimal_to_string(weight_decay), reduction_factor)
    else:
        model_name = "..//..//..//Models//Final_100522//vit_lr{}_wd{}_1on{}.pth".format(decimal_to_string(learning_rate),
                                                                                    decimal_to_string(weight_decay),
                                                                                    reduction_factor)

    print(model_name)

    superclasses = get_superclasses()
    classes = get_classes()
    # given the list of superclasses, returns the class to exclude and the coarse label
    n_classes = len(classes[0])
    n_superclasses = len(superclasses)

    # given the list of superclasses, returns the class to exclude and the coarse label
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)

    classes.append(excluded)
    train_dataset = ClassSpecificImageFolderNotAlphabetic(train_dir, all_dropped_classes=classes, transform=transform)

    test_dataset = ClassSpecificImageFolderNotAlphabetic(test_dir, all_dropped_classes=classes, transform=transform)

    dataset = train_val_dataset(train_dataset, 0.15, reduction_factor)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    num_ftrs = model.classifier.in_features  # input features for the last layers
    model.classifier = nn.Linear(num_ftrs, out_features=n_classes)  # we have 2 classes now
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)

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

                    outputs = model(images).logits
                    _, preds = torch.max(outputs, 1)

                    loss, loss_dict = hierarchical_cc(outputs, labels, np.asarray(coarse_labels),
                                                      return_tree_CIFAR(), int(n_classes / n_superclasses),
                                                      n_superclasses, model, w0, device, hierarchical_loss,
                                                      regularization, sp_regularization, weight_decay)

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

        output = model(inputs).logits  # Feed Network

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