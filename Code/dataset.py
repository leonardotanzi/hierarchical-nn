from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import numpy as np
import cv2
import shutil


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class ImbalanceCIFAR10(datasets.CIFAR10):
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
    def __init__(self, root, all_classes, dropped_classes=[], transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, is_valid_file=None):
        self.dropped_classes = dropped_classes
        self.all_classes = all_classes
        super(ClassSpecificImageFolderNotAlphabetic, self).__init__(root, loader,
                                                                    IMG_EXTENSIONS if is_valid_file is None else None,
                                                                    transform=transform,
                                                                    target_transform=target_transform,
                                                                    is_valid_file=is_valid_file)
        self.imgs = self.samples

    def find_classes(self, directory):
        classes = self.all_classes
        classes = [c for c in classes if c not in self.dropped_classes]
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


def train_val_dataset(dataset, val_split, reduction_factor=1, reduce_val=False, reduction_factor_val=16):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, shuffle=True)
    datasets = {}

    train_idx = [index for i, index in enumerate(train_idx) if i % reduction_factor == 0]
    if reduce_val:
        val_idx = [index for i, index in enumerate(val_idx) if i % reduction_factor_val == 0]

    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets


def build_mapping_imagenet():

    with open("..//..//Dataset//map_clsloc.txt", "r") as f:
        all_mappings = f.readlines()
        dict_code_number = {}
        for mapping in all_mappings:
            code = mapping.split(" ")[0]
            number = mapping.split(" ")[1]
            dict_code_number[code] = number
    with open("..//..//Dataset//gt_valid.txt", "r") as f:
        all_lines = f.readlines()
        dict_img = {}
        for line in all_lines:
            mapping = line.split(",")[0].split("/")
            code = mapping[-1].split("_")[0]
            name = mapping[-2]
            if not os.path.exists(f"..//..//Dataset//Imagenet_leaves//{name}"):
                os.makedirs(f"..//..//Dataset//Imagenet_leaves//{name}")
            dict_img[dict_code_number[code]] = name

    return dict_img


def build_imagenet():
    d = np.load("..//..//Dataset//nparray//Imagenet64_val_npz//Imagenet64_val_npz//val_data.npz")
    x = d['data']
    y = d['labels']
    # y = [i - 1 for i in y]
    img_size = 64 * 64
    x = np.dstack((x[:, :img_size], x[:, img_size:2 * img_size], x[:, 2 * img_size:]))
    x = x.reshape((x.shape[0], 64, 64, 3))

    dict_img = build_mapping_imagenet()

    for i, (img, label) in enumerate(zip(x, y)):
        out_name = f"..//..//Dataset//Imagenet_leaves//{dict_img[str(label)]}//{dict_img[str(label)]}_{i}.png"
        cv2.imwrite(out_name, img)

    for i in range(5):
        d = np.load(f"..//..//Dataset//nparray//Imagenet64_train_part1_npz//Imagenet64_train_part1_npz//train_data_batch_{i + 1}.npz")
        x = d['data']
        y = d['labels']
        # y = [i - 1 for i in y]
        img_size = 64 * 64
        x = np.dstack((x[:, :img_size], x[:, img_size:2 * img_size], x[:, 2 * img_size:]))
        x = x.reshape((x.shape[0], 64, 64, 3))

        dict_img = build_mapping_imagenet()

        for i, (img, label) in enumerate(zip(x, y)):
            out_name = f"..//..//Dataset//Imagenet_leaves//{dict_img[str(label)]}//{dict_img[str(label)]}_{i}.png"
            cv2.imwrite(out_name, img)

    for i in range(5):
        d = np.load(f"..//..//Dataset//nparray//Imagenet64_train_part2_npz//Imagenet64_train_part2_npz//train_data_batch_{i + 1 + 5}.npz")
        x = d['data']
        y = d['labels']
        # y = [i - 1 for i in y]
        img_size = 64 * 64
        x = np.dstack((x[:, :img_size], x[:, img_size:2 * img_size], x[:, 2 * img_size:]))
        x = x.reshape((x.shape[0], 64, 64, 3))

        dict_img = build_mapping_imagenet()

        for i, (img, label) in enumerate(zip(x, y)):
            out_name = f"..//..//Dataset//Imagenet_leaves//{dict_img[str(label)]}//{dict_img[str(label)]}_{i}.png"
            cv2.imwrite(out_name, img)


def build_fgvc(set):

    with open(f"..//..//Dataset//fgvc-aircraft-2013b//data//images_variant_{set}.txt", "r") as file:

        for line in file.readlines():
            token0 = line.split(" ")[0]
            token1 = line[len(token0) + 1:-1]

            if not os.path.exists(f"..//..//Dataset//Aircraft//{set}//{token1}"):
                os.makedirs(f"..//..//Dataset//Aircraft//{set}//{token1}")

            shutil.copy(f"..//..//Dataset//fgvc-aircraft-2013b//data//images//{token0}.jpg",
                        f"..//..//Dataset//Aircraft//{set}//{token1}//{token0}.jpg")