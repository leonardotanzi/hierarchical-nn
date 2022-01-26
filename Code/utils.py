from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


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
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def sparse2coarse(targets, coarse_labels):
    # this is the list of the supergorup to which each class belong (so class 1 belong to superclass 4, classe 2 to superclass 1 and so on)
    return coarse_labels[targets]


def exclude_classes(n_superclasses):
    superclass = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                  ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                  ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                  ['bottle', 'bowl', 'can', 'cup', 'plate'],
                  ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                  ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                  ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                  ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                  ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                  ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                  ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                  ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                  ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                  ['crab', 'lobster', 'snail', 'spider', 'worm'],
                  ['baby', 'boy', 'girl', 'man', 'woman'],
                  ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                  ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                  ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                  ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                  ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
    excluded = []
    coarse_labels = []
    for i, sc in enumerate(superclass):
        if i > (n_superclasses - 1):
            for j in sc:
                excluded.append(j)
        else:
            for j in sc:
                coarse_labels.append(i)

    return excluded, coarse_labels
