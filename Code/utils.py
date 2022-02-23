from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import torch
import cv2


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


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


class ClassSpecificImageFolderNotAlphabetic(datasets.DatasetFolder):
    def __init__(
            self,
            root,
            all_dropped_classes=[],
            transform=None,
            target_transform=None,
            loader=datasets.folder.default_loader,
            is_valid_file=None):
        self.all_dropped_classes = all_dropped_classes
        super(ClassSpecificImageFolderNotAlphabetic, self).__init__(root, loader,
                                                                    IMG_EXTENSIONS if is_valid_file is None else None,
                                                                    transform=transform,
                                                                    target_transform=target_transform,
                                                                    is_valid_file=is_valid_file)
        self.imgs = self.samples

    def find_classes(self, directory):
        classes = self.all_dropped_classes[0]
        classes = [c for c in classes if c not in self.all_dropped_classes[1]]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def get_classes():
    return [['beaver', 'dolphin', 'otter', 'seal', 'whale',
             'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
             'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
             'bottle', 'bowl', 'can', 'cup', 'plate',
             'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
             'clock', 'keyboard', 'lamp', 'telephone', 'television',
             'bed', 'chair', 'couch', 'table', 'wardrobe',
             'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
             'bear', 'leopard', 'lion', 'tiger', 'wolf',
             'bridge', 'castle', 'house', 'road', 'skyscraper',
             'cloud', 'forest', 'mountain', 'plain', 'sea',
             'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
             'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
             'crab', 'lobster', 'snail', 'spider', 'worm',
             'baby', 'boy', 'girl', 'man', 'woman',
             'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
             'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
             'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
             'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
             'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]


def get_superclasses():
    return ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
            'household electrical devices', 'household furniture','insects',
            'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes',
            'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
            'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']


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


def sparse2coarse_full(targets):
    # this is the list of the supergorup to which each class belong (so class 1 belong to superclass 4, classe 2 to superclass 1 and so on)
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


def sparse2coarse(targets, coarse_labels):
    # this is the list of the supergorup to which each class belong (so class 1 belong to superclass 4, classe 2 to superclass 1 and so on)
    return coarse_labels[targets]


def exclude_classes(superclasses_names):
    # superclass = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    #               ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    #               ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    #               ['bottle', 'bowl', 'can', 'cup', 'plate'],
    #               ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    #               ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    #               ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    #               ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    #               ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    #               ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    #               ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    #               ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    #               ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    #               ['crab', 'lobster', 'snail', 'spider', 'worm'],
    #               ['baby', 'boy', 'girl', 'man', 'woman'],
    #               ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    #               ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    #               ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    #               ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    #               ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
    superclass_dict = {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                       'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                       'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                       'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
                       'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                       'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                       'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                       'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                       'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                       'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                       'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                       'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                       'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                       'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
                       'people': ['baby', 'boy', 'girl', 'man', 'woman'],
                       'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                       'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                       'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                       'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                       'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}

    excluded = []
    coarse_labels = []

    for i, superclass in enumerate(superclass_dict):
        if superclass not in superclasses_names:
            for fine_class in superclass_dict[superclass]:
                excluded.append(fine_class)

    for i in range(len(superclasses_names)):
        tmp = [i] * 5
        coarse_labels.extend(tmp)

    return excluded, coarse_labels


def accuracy_superclasses(predicted, actual, coarse_labels, n_superclass):
    batch = predicted.size(0)
    predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device="cuda:0")

    for k in range(n_superclass):
        # obtain the indexes of the superclass number k
        indexes = list(np.where(coarse_labels == k))[0]
        # for each index, sum all the probability related to that superclass
        for j in indexes:
            predicted_coarse[:, k] = predicted_coarse[:, k] + predicted[:, j]

    actual_coarse = sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)
    predicted_coarse = np.argmax(predicted_coarse.cpu().detach().numpy(), axis=1)

    acc = np.sum(np.equal(actual_coarse, predicted_coarse)) / len(actual_coarse)
    return acc


def to_latex_heatmap(n_classes, classes_name, matrix):

    # "\newcommand\items{4}   %Number of classes
    # \arrayrulecolor{black} %Table line colors
    # \noindent\begin{tabular}{c*{\items}{|E}|}

    # this output

    # \end{tabular}"

    basic_string = "\multicolumn{1}{c}{} &" + "\n"

    for i in range(n_classes):
        basic_string += "\multicolumn{1}{c}{" + str(i + 1) + "} "
        if i != n_classes - 1:
            basic_string += "& \n"

    basic_string += "\\\ \hhline{~ *\items{ | -} |}" + "\n"

    for i in range(n_classes):
        basic_string += str(i + 1)

        for j in range(n_classes):
            basic_string += "& " + f"{matrix[i][j]:.1f}"

        basic_string += " \\\ \hhline{~*\items{|-}|}" + "\n"

    print(basic_string)

#
# A  & 100   & 0  & 10  & 0   \\ \hhline{~*\items{|-}|}
# B  & 10   & 80  & 10  & 0 \\ \hhline{~*\items{|-}|}
# C  & 30   & 0   & 70  & 0 \\ \hhline{~*\items{|-}|}
# D  & 30   & 0   & 70  & 0 \\ \hhline{~*\items{|-}|}
# \end{tabular}"


def readpgm(name):

    with open(name) as f:

        lines = f.readlines()

    # This ignores commented lines

    for l in list(lines):

        if l[0] == '#':

            lines.remove(l)

    # here,it makes sure it is ASCII format (P2)

    assert lines[0].strip() == 'P2'

    # Converts data to a list of integers

    data = []

    for line in lines[1:]:

        data.extend([int(c) for c in line.split()])

    return (np.array(data[3:]),(data[1],data[0]),data[2])


if __name__ == "__main__":

    to_latex_heatmap(3, ["a", "b", "c"], [[411, 75, 14], [53, 436, 11], [3,  28, 469]])
