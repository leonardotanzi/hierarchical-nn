import torch


def decimal_to_string(dec):
    str_old = str(dec)
    str_new = ""
    for i in str_old:
        if i != ".":
            str_new += i
    return str_new


def get_medium_labels(superclasses_names):
    medium_labels = []

    for i in range(len(superclasses_names)):
        tmp = [i] * 5
        medium_labels.extend(tmp)

    return medium_labels


def get_coarse_labels(superclasses_names):
    coarse_labels = []
    occurencies = [2, 5, 2, 3, 3, 2, 2, 1]
    occurencies = [i * 5 for i in occurencies]

    for i in range(len(superclasses_names)):
        tmp = [i] * occurencies[i]
        coarse_labels.extend(tmp)

    return coarse_labels


def sparser2coarser(targets, coarser_labels):
    # this is the list of the supergorup to which each class belong (so class 1 belong to superclass 4, classe 2 to superclass 1 and so on)
    return coarser_labels[targets]


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
    return ['beaver', 'dolphin', 'otter', 'seal', 'whale',
            'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',

            'bear', 'leopard', 'lion', 'tiger', 'wolf',
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',

            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            'crab', 'lobster', 'snail', 'spider', 'worm',

            'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
            'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',

            'bottle', 'bowl', 'can', 'cup', 'plate',
            'clock', 'keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',

            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea',

            'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
            'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor',

            'baby', 'boy', 'girl', 'man', 'woman']


def get_superclasses():
    return ['aquatic mammals', 'fish',
            'large carnivores', 'large omnivores and herbivores', 'medium-sized mammals', 'reptiles', 'small mammals',
            'insects', 'non-insect invertebrates',
            'flowers', 'fruit and vegetables', 'trees',
            'food containers', 'household electrical devices', 'household furniture',
            'large man-made outdoor things', 'large natural outdoor scenes',
            'vehicles 1', 'vehicles 2',
            'people']


def get_hyperclasses():
    return ['sea animal', 'land animal', 'insect and invertebrates', 'flora', 'objects', 'outdoor scene', 'vehicles',
            'people']


def build_cifar_matrix():
    # build one dict for each layer
    first_dict = {'sea animal': ['aquatic mammals', 'fish'],
                  'land animal': ['large carnivores', 'large omnivores and herbivores', 'medium-sized mammals', 'reptiles', 'small mammals'],
                  'insect and invertebrates': ['insects', 'non-insect invertebrates'],
                  'flora': ['flowers', 'fruit and vegetables', 'trees'],
                  'objects': ['food containers', 'household electrical devices', 'household furniture'],
                  'outdoor scenes': ['large man-made outdoor things', 'large natural outdoor scenes'],
                  'vehicles': ['vehicles 1', 'vehicles 2'],
                  'people': ['people']
                  }

    second_dict = {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
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

    all_dict = [first_dict, second_dict]


if __name__ == "__main__":
    # to_latex_heatmap(3, ["a", "b", "c"], [[411, 75, 14], [53, 436, 11], [3,  28, 469]])
    # return_tree_CIFAR()
    pass
