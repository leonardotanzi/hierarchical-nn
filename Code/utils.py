from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import torch
import torch.nn.functional as F
import copy
from anytree import Node, RenderTree, LevelOrderGroupIter


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def hierarchical_cc(predicted, actual, coarse_labels, tree, n_class, n_superclass, model, w0, device, hierarchical_loss,
                    regularization, sp_regularization, weight_decay):

    batch = predicted.size(0)
    # compute the loss for fine classes
    loss = F.cross_entropy(predicted, actual, reduction="mean")

    if hierarchical_loss:
        # define an empty vector which contains 20 superclasses prediction for each samples
        predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device=device)

        for k in range(n_superclass):
            # obtain the indexes of the superclass number k
            indexes = list(np.where(coarse_labels == k))[0]
            # for each index, sum all the probability related to that superclass
            # for each line, at the position k, you sum all the classe related to superclass k, so for k=0 the classes are 0 to 4
            predicted_coarse[:, k] += torch.sum(predicted[:, indexes], dim=1)
            # this line is like the cycle below but more fast
            # for j in indexes:
            #     predicted_coarse[:, k] = predicted_coarse[:, k] + predicted[:, j]

        actual_coarse = sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)

        loss_coarse = F.cross_entropy(predicted_coarse, torch.from_numpy(actual_coarse).type(torch.int64).to(device), reduction="mean")
        loss += loss_coarse

    if regularization:
        # beta = model.fc.weight.data
        #
        # penalty = 0
        # for i, node in enumerate(LevelOrderGroupIter(tree)):
        #     # start computing not for root but for first layer
        #     if i > 0:
        #         # iniziare a ciclare sul primo livello (nel caso cifar, superclassi)
        #         for level_class in node:
        #             # se sopra di loro c'è root non ho ancestor (quindi il secondo termine nell'equazione è =0
        #             n_ancestors = 0 if i == 1 else 1
        #             # prendo tutte le foglie del nodo (root avrà 100, una superclasse ne ha 5)
        #             descendants = level_class.leaves
        #             # PRIMO TERMINE
        #             # se sono allultimo livello (quello delle classi, dove la heigth è zero,
        #             # la formula è beta - mean(beta_parent), quindi devo prendere un solo vettore dai pesi
        #             # come primo termine
        #             if level_class.height == 0:
        #                 position = class_to_index(level_class.name)
        #                 beta_vec_node = model.fc.weight.data[position]
        #             # se sono in un altro livello vado invece a prendere tutti i beta relativi alle leaf
        #             else:
        #                 for j, classes_name in enumerate(descendants):
        #                     # recupero l'indice associato al nome della classe
        #                     position = class_to_index(classes_name.name)
        #                     # prendo il vettore tra i pesi relativo a quell'indice
        #                     beta_vec_node = model.fc.weight.data[position][None, :] if j == 0 else torch.cat((beta_vec_node, model.fc.weight.data[position][None, :]), 0)
        #             # SECONDO TERMINE
        #             # I have to do the same thing but this time with the leaves of the parent
        #             for k, superclasses_name in enumerate(level_class.ancestors[i-1].leaves):
        #                 position = class_to_index(superclasses_name.name)
        #                 beta_vec_parent = model.fc.weight.data[position][None, :] if k == 0 else torch.cat((beta_vec_parent, model.fc.weight.data[position][None, :]), 0)
        #
        #             # se n_ancestor è zero significa che il secondo termine non c'è, è il caso del primo livello
        #             penalty += torch.linalg.norm(len(descendants) * (torch.mean(beta_vec_node, dim=0) - n_ancestors*(torch.mean(beta_vec_parent, dim=0))))
        #
        # print(penalty)
        #
        # loss += weight_decay * penalty

        # sommo le mean per le superclassi e le ripeto cosi ho un array delle stesse dim [15, 2048] e i primi cinque sono la mean delle prime cinque clasi,
        # secondi cinque delle seconde cinque classi e cosi via

        for i in range(n_superclass):
            if i == 0:
                mean_betas = torch.mean(model.fc.weight.data[i*n_class:i*n_class + n_class], dim=0, keepdim=True).repeat(n_class, 1)
            else:
                mean_betas = torch.cat((mean_betas, torch.mean(model.fc.weight.data[i*n_class:i*n_class + n_class], dim=0, keepdim=True).repeat(n_class, 1)), 0)

        fine_penalty = torch.sum(torch.linalg.norm(model.fc.weight.data - mean_betas, dim=1))

        for i in range(n_superclass):
            if i == 0:
                coarse_penalty = torch.linalg.norm(torch.sum(model.fc.weight.data[i:i+n_class], dim=0))
            else:
                coarse_penalty += torch.linalg.norm(5*model.fc.weight.data[i:i + n_class])

        print(fine_penalty + coarse_penalty)
        loss += weight_decay * (fine_penalty + coarse_penalty)

    if sp_regularization:
        w = []
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'weight' in name and 'fc' not in name:
                w.append(W.view(-1))
        l2_reg = torch.linalg.norm(torch.cat(w) - w0)
        loss += weight_decay * l2_reg

    return loss


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


class class_specific_image_folder_not_alphabetic(datasets.DatasetFolder):
    def __init__(
            self,
            root,
            all_dropped_classes=[],
            transform=None,
            target_transform=None,
            loader=datasets.folder.default_loader,
            is_valid_file=None):
        self.all_dropped_classes = all_dropped_classes
        super(class_specific_image_folder_not_alphabetic, self).__init__(root, loader,
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

def class_to_index(token):
    classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
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
             'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    return classes.index(token)


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
        predicted_coarse[:, k] += torch.sum(predicted[:, indexes], dim=1)

    actual_coarse = sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)
    predicted_coarse = np.argmax(predicted_coarse.cpu().detach().numpy(), axis=1)
    running_corrects = np.sum(predicted_coarse == actual_coarse)

    return running_corrects


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


def return_tree_CIFAR():
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

    root = Node("root")
    for i, (k, v) in enumerate(superclass_dict.items()):
        p = Node(f"{k}", parent=root)
        for c in v:
            n = Node(f"{c}", parent=p)
    return root


if __name__ == "__main__":

    # to_latex_heatmap(3, ["a", "b", "c"], [[411, 75, 14], [53, 436, 11], [3,  28, 469]])
    return_tree_CIFAR()