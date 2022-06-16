from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import torchvision
import numpy as np


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 classes=None):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.classes = classes
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalanceCIFAR100(ImbalanceCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [['train', '16019d7e3df5f24257cddd939b257f8d']]
    test_list = [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc']]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class ClassSpecificImageFolderNotAlphabetic(datasets.DatasetFolder):
    def __init__(self, root, all_dropped_classes=[], transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, is_valid_file=None):
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


class ImageFolderNotAlphabetic(datasets.DatasetFolder):
    def __init__(self, root, classes, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, is_valid_file=None):
        self.classes = classes
        super(ImageFolderNotAlphabetic, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                    transform=transform,
                                                    target_transform=target_transform,
                                                    is_valid_file=is_valid_file)
        self.imgs = self.samples

    def find_classes(self, directory):
        classes = self.classes
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def train_val_dataset(dataset, val_split, reduction_factor=1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, shuffle=True)
    datasets = {}

    train_idx = [index for i, index in enumerate(train_idx) if i % reduction_factor == 0]
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets


def exclude_classes(superclasses_names):
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


if __name__ == "__main__":
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = IMBALANCECIFAR100(root='./data', train=True,
    #                              download=True, transform=transform)
    # trainloader = iter(trainset)
    # data, label = next(trainloader)
