import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from utils import get_classes, get_superclasses, ClassSpecificImageFolderNotAlphabetic, train_val_dataset, \
    exclude_classes, sparse2coarse
import torch.nn.functional as F
from torchsummary import summary
import math


class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()

        self.mhsa = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.mlp_exp = nn.Linear(in_features=512, out_features=1024)
        self.mlp_compr = nn.Linear(in_features=1024, out_features=512)

    def forward(self, x):
        x_mhsa, _ = self.mhsa(x, x, x)
        x_mlp = self.mlp_exp(x_mhsa)
        x_mlp = self.mlp_compr(x_mlp)
        return x_mlp


class CNNEncoder(nn.Module):
    def __init__(self, n_superclasses, n_classes, input_size):
        super(CNNEncoder, self).__init__()

        self.first_view = 128 * (input_size // 8) * (input_size // 8)
        self.second_view = 1024 * (input_size // 64) * (input_size // 64)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding="same")
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same")

        self.pool = nn.MaxPool2d(2, 2)

        self.pre_fc_super = nn.Linear(in_features=self.first_view, out_features=1024)
        self.fc_super = nn.Linear(in_features=1024, out_features=n_superclasses)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same")
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same")
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding="same")

        self.pre_fc = nn.Linear(in_features=self.second_view, out_features=1024)
        self.fc = nn.Linear(in_features=1024, out_features=n_classes)

        ########################################

        self.proj1 = nn.Linear(in_features=32 * 64 * 64, out_features=512)
        self.proj2 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.proj3 = nn.Linear(in_features=128 * 16 * 16, out_features=512)

        self.proj4 = nn.Linear(in_features=256 * 8 * 8, out_features=512)
        self.proj5 = nn.Linear(in_features=512 * 4 * 4, out_features=512)
        self.proj6 = nn.Linear(in_features=1024 * 2 * 2, out_features=512)

        self.transf = TransformerBlock()
        self.final_cl = nn.Linear(in_features=512*6, out_features=n_classes)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        token1 = self.proj1(x.view(-1, 32 * 64 * 64))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        token2 = self.proj2(x.view(-1, 64 * 32 * 32))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        token3 = self.proj3(x.view(-1, 128 * 16 * 16))

        xfcs = x.view(-1, self.first_view)
        xfcs = torch.relu(self.pre_fc_super(xfcs))
        xfcs = self.fc_super(xfcs)

        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        token4 = self.proj4(x.view(-1, 256 * 8 * 8))
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        token5 = self.proj5(x.view(-1, 512 * 4 * 4))
        x = torch.relu(self.conv6(x))
        x = self.pool(x)
        token6 = self.proj6(x.view(-1, 1024 * 2 * 2))

        xfc = x.view(-1, self.second_view)
        xfc = torch.relu(self.pre_fc(xfc))
        xfc = self.fc(xfc)

        x = torch.cat([token1, token2, token3, token4, token5, token6], dim=1)
        x = x.view(-1, 6, 512)
        x = self.transf(x)
        x = x.view(-1, 512*6)
        x = self.final_cl(x)

        return xfcs, xfc, x

# class SAFModel(nn.Module):
def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    s = torch.matmul(q, torch.transpose(k, 0, 1))
    s /= math.sqrt(d_k)
    attention = F.softmax(s, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


if __name__ == "__main__":

    device = "cuda:0"
    batch_size = 128
    validation_split = 0.15
    n_epochs = 100
    input_size = 128

    train_dir = "..//..//cifar//train//"
    test_dir = "..//..//cifar//test//"
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((input_size, input_size))])

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

    model = CNNEncoder(n_superclasses, n_classes, input_size)
    model.to(device)

    print(summary(model, (3, 128, 128)))

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")

        for phase in ["train", "val"]:

            running_loss = 0.0
            running_acc = 0.0
            running_superacc = 0.0
            running_attention_acc = 0.0

            if phase is "train":
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

                    loss = 0.5*(loss_superclass + loss_class) + 0.5*loss_attention

                    _, class_preds = torch.max(class_output, 1)
                    _, superclass_preds = torch.max(superclass_output, 1)
                    _, attention_preds = torch.max(attention_output, 1)

                    if phase is "train":
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
            epoch_loss = running_loss / dataset_sizes[phase]

            print(f"{phase}: loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}, super accuracy {epoch_superacc:.4f}, attention accuracy {epoch_attention:.4f}")






