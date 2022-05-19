import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
from utils import ClassSpecificImageFolderNotAlphabetic, exclude_classes, get_classes
from OldCode.CNN_CIFAR100 import ConvNet

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_size = 32
    batch_size = 128

    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dir = "../../../cifar/test//"

    classes_name = get_classes()
    superclasses = ["flowers", "fruit and vegetables", "trees"]
    # superclasses = get_superclasses()

    excluded, coarse_labels = exclude_classes(superclasses)

    # build a list where the first elements are all the classes and the second the class to exclude
    classes_name.append(excluded)

    test_dataset = ClassSpecificImageFolderNotAlphabetic(test_dir, all_dropped_classes=classes_name, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_names = test_dataset.classes

    for i in range(5):
        best_model_wts = "..\\..\\Models\\cnn_hierarchical_regularization\\Fold{}_cnn_hierarchical_regularization.pth".format(i+1)

        model = ConvNet(num_classes=len(class_names))

        model.load_state_dict(torch.load(best_model_wts))

        model.to(device)

        model.eval()

        y_pred = []
        y_true = []

        # iterate over test data
        max_acc = 0.0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        acc = np.sum(np.equal(y_true, y_pred)) / len(y_true)
        if acc > max_acc:
            max_acc = acc
            max_acc_model = model
            selected_fold = i + 1
        print("Fold {}, accuracy {}".format(i + 1, acc))
        print(classification_report(y_true, y_pred))

    # Build confusion matrix
    print("Best accuracy with Fold {}".format(selected_fold))
    model = max_acc_model
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(class_names), index=[i for i in class_names], columns=[i for i in class_names])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f")
    plt.savefig("..\\ConfusionMatrixes\\{}-CMpercentage.png".format(best_model_wts.split("\\")[-1].split(".")[0]))

    df_cm = pd.DataFrame(cf_matrix, index=[i for i in class_names], columns=[i for i in class_names])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, cmap="coolwarm", annot=True)
    plt.savefig("..\\ConfusionMatrixes\\{}-CM.png".format(best_model_wts.split("\\")[-1].split(".")[0]))

    number_of_superclasses = len(superclasses)
    classes_for_superclass = 5
    cf_matrix_super = np.zeros(shape=(number_of_superclasses, number_of_superclasses))

    for row in range(number_of_superclasses):
        for col in range(number_of_superclasses):
            cf_matrix_super[row, col] = np.sum(cf_matrix[row * classes_for_superclass:
                                                         row * classes_for_superclass + classes_for_superclass,
                                               col * classes_for_superclass: col * classes_for_superclass + classes_for_superclass])

    df_cm = pd.DataFrame(cf_matrix_super, index=[i for i in superclasses], columns=[i for i in superclasses])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt="g")
    plt.savefig("..\\ConfusionMatrixes\\{}-CMsuperclass.png".format(best_model_wts.split("\\")[-1].split(".")[0]))

    df_cm = pd.DataFrame(cf_matrix_super / np.sum(cf_matrix_super) * number_of_superclasses, index=[i for i in superclasses],
                         columns=[i for i in superclasses])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt=".2f")
    plt.savefig("..\\ConfusionMatrixes\\{}-CMsuperclass_percentage.png".format(best_model_wts.split("\\")[-1].split(".")[0]))