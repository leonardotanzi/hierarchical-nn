import matplotlib.pyplot as plt
import numpy as np
import torch
from anytree import Node, RenderTree, LevelOrderGroupIter
import pickle


def save_list(file_name, list_to_save):
    open_file = open(file_name, "wb")
    pickle.dump(list_to_save, open_file)
    open_file.close()


def load_list(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list


def decimal_to_string(dec):
    str_old = str(dec)
    str_new = ""
    for i in str_old:
        if i != ".":
            str_new += i
    return str_new


def sparse2coarse(targets, coarse_labels):
    # this is the list of the supergorup to which each class belong (so class 1 belong to superclass 4, classe 2 to superclass 1 and so on)
    return coarse_labels[targets]


def evaluate_regularization(model, n_class=5, n_superclass=20):
    coarse_penalty = 0.0
    fine_penalty = 0.0
    for i in range(n_superclass):
        coarse_penalty += (torch.linalg.norm(
            torch.sum(model.fc.weight.data[i * n_class:i * n_class + n_class], dim=0))) ** 2
    for i in range(n_class * n_superclass):
        sc_index = 1 // 5
        fine_penalty += (torch.linalg.norm(model.fc.weight.data[i] - 1 / n_class * torch.sum(
            model.fc.weight.data[sc_index * n_class:sc_index * n_class + n_class]))) ** 2

    print(coarse_penalty)
    print(fine_penalty)


def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]




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
            'household electrical devices', 'household furniture', 'insects',
            'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes',
            'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
            'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def accuracy_superclasses(predicted, actual, coarse_labels, n_superclass, device):
    batch = predicted.size(0)
    predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device="cuda:0")

    for k in range(n_superclass):
        # obtain the indexes of the superclass number k
        indexes = list(np.where(coarse_labels == k))[0]
        predicted_coarse[:, k] += torch.sum(predicted[:, indexes], dim=1)

    coarse_labels = torch.tensor(coarse_labels).type(torch.int64).to(device)
    actual_coarse = sparse2coarse(actual, coarse_labels)

    # actual_coarse = sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)
    predicted_coarse = torch.argmax(predicted_coarse, dim=1)
    running_corrects = torch.sum(predicted_coarse == actual_coarse)

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
    return (np.array(data[3:]), (data[1], data[0]), data[2])


def return_tree_CIFAR(reduced=False):
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

    if reduced:
        superclass_dict = {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                           'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']}

    root = Node("root")
    for i, (k, v) in enumerate(superclass_dict.items()):
        p = Node(f"{k}", parent=root)
        for c in v:
            n = Node(f"{c}", parent=p)
    return root


if __name__ == "__main__":
    # to_latex_heatmap(3, ["a", "b", "c"], [[411, 75, 14], [53, 436, 11], [3,  28, 469]])
    # return_tree_CIFAR()
    pass
