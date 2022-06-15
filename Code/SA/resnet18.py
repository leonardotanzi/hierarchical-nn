import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from utils import get_classes, get_superclasses, ClassSpecificImageFolderNotAlphabetic, train_val_dataset, \
    exclude_classes, sparse2coarse
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(ResBlock, self).__init__()

        if not downsample:
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.skip = nn.Sequential()
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2)),
                nn.BatchNorm2d(out_channels)
            )

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        shortcut = self.skip(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        return torch.relu(x + shortcut)


class ResNet18(nn.Module):
    def __init__(self, n_classes):
        super(ResNet18, self).__init__()

        self.layer0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.pool0 = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1_1 = ResBlock(64, 64, False)
        self.conv1_2 = ResBlock(64, 64, False)

        self.conv2_1 = ResBlock(64, 128, True)
        self.conv2_2 = ResBlock(128, 128, False)

        self.conv3_1 = ResBlock(128, 256, True)
        self.conv3_2 = ResBlock(256, 256, False)

        self.conv4_1 = ResBlock(256, 512, True)
        self.conv4_2 = ResBlock(512, 512, False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):

        x = torch.relu(self.bn0(self.layer0(x)))
        x = self.pool0(x)

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":

    device = "cuda:0"
    batch_size = 128
    validation_split = 0.15
    n_epochs = 100
    input_size = 224

    train_dir = "..//..//cifar//train//"
    test_dir = "..//..//cifar//test//"

    transform = Compose(
        [ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((input_size, input_size))])

    classes = get_classes()
    superclasses = get_superclasses()
    n_classes = len(classes[0])
    n_superclasses = len(superclasses)
    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)
    classes.append(excluded)

    train_dataset = ClassSpecificImageFolderNotAlphabetic(train_dir, all_dropped_classes=classes, transform=transform)
    dataset = train_val_dataset(train_dataset, validation_split)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    model = ResNet18(n_classes)
    model.to(device)

    # print(summary(model, (3, 128, 128)))

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        for phase in ["train", "val"]:

            running_loss = 0.0
            running_acc = 0.0
            running_superacc = 0.0
            running_attention_acc = 0.0

            if phase == "train":
                loader = train_loader
            else:
                loader = val_loader

            for (images, class_labels) in loader:

                images = images.to(device)
                class_labels = class_labels.to(device)

                with torch.set_grad_enabled(phase == "train"):

                    output = model(images)

                    loss = F.cross_entropy(output, class_labels)

                    _, class_preds = torch.max(output, 1)

                    if phase == "train":
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                running_loss += loss.item()
                running_acc += torch.sum(class_preds == class_labels.data).item()

            epoch_acc = running_acc / dataset_sizes[phase]
            epoch_loss = running_loss / len(loader)

            print(
                f"{phase}: loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

