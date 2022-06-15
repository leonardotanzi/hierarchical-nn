import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from utils import get_classes, get_superclasses, , sparse2coarse
from dataset import ClassSpecificImageFolderNotAlphabetic, train_val_dataset, exclude_classes
import torch.nn.functional as F
from features_self_attention import TransformerBlock


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


class ResNet18_MHSA(nn.Module):
    def __init__(self, n_superclasses, n_classes, seq_len, d_model, heads):
        super(ResNet18_MHSA, self).__init__()

        self.first_view = 28*28*128
        self.second_view = 7*7*512
        self.d_model = d_model
        self.seq_len = seq_len
        self.heads = heads

        self.layer0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.pool0 = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1_1 = ResBlock(64, 64, False)
        self.conv1_2 = ResBlock(64, 64, False)

        self.conv2_1 = ResBlock(64, 128, True)
        self.conv2_2 = ResBlock(128, 128, False)

        self.pre_fc_super = nn.Linear(in_features=self.first_view, out_features=1024)
        self.fc_super = nn.Linear(in_features=1024, out_features=n_superclasses)

        self.conv3_1 = ResBlock(128, 256, True)
        self.conv3_2 = ResBlock(256, 256, False)

        self.conv4_1 = ResBlock(256, 512, True)
        self.conv4_2 = ResBlock(512, 512, False)

        self.pre_fc = nn.Linear(in_features=self.second_view, out_features=1024)
        self.fc = nn.Linear(in_features=1024, out_features=n_classes)

        self.proj1 = nn.Linear(in_features=64 * 56 * 56, out_features=d_model)
        self.proj2 = nn.Linear(in_features=128 * 28 * 28, out_features=d_model)
        self.proj3 = nn.Linear(in_features=256 * 14 * 14, out_features=d_model)
        self.proj4 = nn.Linear(in_features=512 * 7 * 7, out_features=d_model)

        self.transformer1 = TransformerBlock(seq_len, d_model, heads, n_classes)
        self.transformer2 = TransformerBlock(seq_len, d_model, heads, n_classes)
        self.transformer3 = TransformerBlock(seq_len, d_model, heads, n_classes)

        self.linear = nn.Linear(seq_len * d_model, n_classes)

    def forward(self, x):

        x = torch.relu(self.bn0(self.layer0(x)))
        x = self.pool0(x)

        x = self.conv1_1(x)
        token1 = self.proj1(x.view(-1, 64 * 56 * 56))
        x = self.conv1_2(x)
        token2 = self.proj1(x.view(-1, 64 * 56 * 56))
        x = self.conv2_1(x)
        token3 = self.proj2(x.view(-1, 128 * 28 * 28))
        x = self.conv2_2(x)
        token4 = self.proj2(x.view(-1, 128 * 28 * 28))

        xfcs = x.view(-1, self.first_view)
        xfcs = torch.relu(self.pre_fc_super(xfcs))
        xfcs = self.fc_super(xfcs)

        x = self.conv3_1(x)
        token5 = self.proj3(x.view(-1, 256 * 14 * 14))
        x = self.conv3_2(x)
        token6 = self.proj3(x.view(-1, 256 * 14 * 14))
        x = self.conv4_1(x)
        token7 = self.proj4(x.view(-1, 512 * 7 * 7))
        x = self.conv4_2(x)
        token8 = self.proj4(x.view(-1, 512 * 7 * 7))

        xfc = x.view(-1, self.second_view)
        xfc = torch.relu(self.pre_fc(xfc))
        xfc = self.fc(xfc)

        x = torch.cat([token1, token2, token3, token4, token5, token6, token7, token8], dim=1)
        x = x.view(-1, self.seq_len, self.d_model)

        x = self.transformer1(x)
        # x = self.transformer2(x)
        # x = self.transformer3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return xfcs, xfc, x


if __name__ == "__main__":

    device = "cuda:0"
    batch_size = 32
    validation_split = 0.15
    n_epochs = 100
    input_size = 224

    d_model = 512
    heads = 1
    seq_len = 8  # n layer conv

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

    model = ResNet18_MHSA(n_superclasses, n_classes, seq_len, d_model, heads)
    model.to(device)

    print(summary(model, (3, 224, 224)))

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

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

                    superclass_output, class_output, attention_output = model(images)

                    coarse_labels = torch.tensor(coarse_labels).type(torch.int64).to(device)
                    superclass_labels = sparse2coarse(class_labels, coarse_labels)

                    loss_superclass = F.cross_entropy(superclass_output, superclass_labels)
                    loss_class = F.cross_entropy(class_output, class_labels)
                    loss_attention = F.cross_entropy(attention_output, class_labels)

                    if epoch < 10:
                        loss = loss_superclass + loss_class
                    else:
                        loss = loss_attention

                    # if epoch < 3:
                    #     loss = loss_superclass
                    # elif 3 < epoch < 6:
                    #     loss = loss_superclass + loss_class
                    # else:
                    #     loss = loss_attention

                    _, class_preds = torch.max(class_output, 1)
                    _, superclass_preds = torch.max(superclass_output, 1)
                    _, attention_preds = torch.max(attention_output, 1)

                    if phase == "train":
                    #     if epoch < 3:
                    #         for name, param in model.named_parameters():
                    #             name = name.split(".")[0]
                    #             if name in ["layer0", "bn0", "conv1_1", "conv1_2", "conv2_1", "conv2_2", "pre_fc_super", "fc_super"]:
                    #                 param.requires_grad = True
                    #             else:
                    #                 param.requires_grad = False
                    #     elif 3 < epoch < 6:
                    #         for name, param in model.named_parameters():
                    #             name = name.split(".")[0]
                    #             if name in ["layer0", "bn0", "conv1_1", "conv1_2", "conv2_1", "conv2_2", "pre_fc_super", "fc_super",
                    #                         "conv3_1", "conv3_2", "conv4_1", "conv4_2", "fc", "pre_fc"]:
                    #                 param.requires_grad = True
                    #             else:
                    #                 param.requires_grad = False
                    #     else:
                    #         for name, param in model.named_parameters():
                    #             name = name.split(".")[0]
                    #             if name in ["layer0", "bn0", "conv1_1", "conv1_2", "conv2_1", "conv2_2", "pre_fc_super", "fc_super",
                    #                         "conv3_1", "conv3_2", "conv4_1", "conv4_2", "fc", "pre_fc"]:
                    #                 param.requires_grad = False
                    #             else:
                    #                 param.requires_grad = True

                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                running_loss += loss.item()
                running_acc += torch.sum(class_preds == class_labels.data).item()
                running_superacc += torch.sum(superclass_preds == superclass_labels.data).item()
                running_attention_acc += torch.sum(attention_preds == class_labels.data).item()

            epoch_acc = running_acc / dataset_sizes[phase]
            epoch_superacc = running_superacc / dataset_sizes[phase]
            epoch_attention = running_attention_acc / dataset_sizes[phase]
            epoch_loss = running_loss / len(loader)

            print(
                f"{phase}: loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}, super accuracy {epoch_superacc:.4f}, attention accuracy {epoch_attention:.4f}")






