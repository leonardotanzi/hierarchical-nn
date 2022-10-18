from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import numpy as np
import cv2
import shutil


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


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