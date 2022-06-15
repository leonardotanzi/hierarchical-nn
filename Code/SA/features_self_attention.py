import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from utils import get_classes, get_superclasses, ClassSpecificImageFolderNotAlphabetic, train_val_dataset, \
    exclude_classes, sparse2coarse
import torch.nn.functional as F
from torchsummary import summary
import math


def attention(q, k, v, dk):

    scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk), dim=-1)  # transpose last two dimension

    output = torch.matmul(scores, v)

    return output, scores


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadSelfAttention, self).__init__()

        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads  # why?

        self.Wq = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.Wk = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.Wv = nn.Linear(in_features=self.d_model, out_features=self.d_model)

        self.W0 = nn.Linear(d_model, d_model)

        self.attention_scores = None

    def forward(self, q, k, v):  # here q, k, v are actually the input embedding, multiplied by W to obtain q,k,v
        batch_size = q.size(0)

        # multiply input for Wq, Wk, Wv to get w, q, v. then reshape including the number of heads
        q = self.Wq(q).view(batch_size, -1, self.heads, self.d_k)
        k = self.Wk(k).view(batch_size, -1, self.heads, self.d_k)
        v = self.Wv(v).view(batch_size, -1, self.heads, self.d_k)

        # transpose to have [batch, heads, seq_len, d_model]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute output and scores for self-attention matrix
        att_output, scores = attention(q, k, v, self.d_k)

        self.attention_scores = scores

        # concatenate heads and put through final linear layer
        concat = att_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W0(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_expansion=2048):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.expansion = nn.Linear(in_features=d_model, out_features=d_expansion)
        self.compression = nn.Linear(in_features=d_expansion, out_features=d_model)

    def forward(self, x):
        x = torch.relu(self.expansion(x))
        x = self.compression(x)  # why no relu here?
        return x


class TransformerBlock(nn.Module):
    def __init__(self, seq_len, d_model, heads, n_classes):
        super(TransformerBlock, self).__init__()
        # self.embedding = nn.Linear(channels, d_model)
        self.attn = MultiHeadSelfAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x_emb = self.embedding(x)
        x_att = self.attn(x, x, x) + x
        x_norm1 = self.norm1(x_att)
        x_ff = self.ff(x_norm1) + x_norm1
        x_norm2 = self.norm2(x_ff)
        return x_norm2


class CNNEncoder(nn.Module):
    def __init__(self, n_superclasses, n_classes, input_size, seq_len, d_model, heads):
        super(CNNEncoder, self).__init__()

        self.first_view = 128 * (input_size // 8) * (input_size // 8)
        self.second_view = 1024 * (input_size // 64) * (input_size // 64)

        self.seq_len = seq_len
        self.d_model = d_model
        self.heads = heads

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

        self.proj1 = nn.Linear(in_features=32 * 64 * 64, out_features=d_model)
        self.proj2 = nn.Linear(in_features=64 * 32 * 32, out_features=d_model)
        self.proj3 = nn.Linear(in_features=128 * 16 * 16, out_features=d_model)

        self.proj4 = nn.Linear(in_features=256 * 8 * 8, out_features=d_model)
        self.proj5 = nn.Linear(in_features=512 * 4 * 4, out_features=d_model)
        self.proj6 = nn.Linear(in_features=1024 * 2 * 2, out_features=d_model)

        self.transformer1 = TransformerBlock(seq_len, d_model, heads, n_classes)
        self.transformer2 = TransformerBlock(seq_len, d_model, heads, n_classes)
        self.transformer3 = TransformerBlock(seq_len, d_model, heads, n_classes)

        self.linear = nn.Linear(seq_len * d_model, n_classes)

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
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return xfcs, xfc, x


if __name__ == "__main__":

    device = "cuda:0"
    batch_size = 128
    validation_split = 0.15
    n_epochs = 100
    input_size = 128

    d_model = 512
    heads = 2
    seq_len = 6  #n layer conv

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

    model = CNNEncoder(n_superclasses, n_classes, input_size, seq_len, d_model, heads)
    model.to(device)

    # print(summary(model, (3, 128, 128)))

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")

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
                        loss = loss_superclass
                    elif 10 < epoch < 20:
                        loss = loss_superclass + loss_class
                    else:
                        loss = loss_attention

                    _, class_preds = torch.max(class_output, 1)
                    _, superclass_preds = torch.max(superclass_output, 1)
                    _, attention_preds = torch.max(attention_output, 1)

                    if phase == "train":
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        if epoch < 10:
                            for name, param in model.named_parameters():
                                name = name.split(".")[0]
                                if name in ["conv1", "conv2", "conv3", "pre_fc_super", "fc_super"]:
                                    param.requires_grad = True
                                else:
                                    param.requires_grad = False
                        elif 10 < epoch < 20:
                            for name, param in model.named_parameters():
                                name = name.split(".")[0]
                                if name in ["conv1", "conv2", "conv3", "pre_fc_super", "fc_super", "conv4", "conv5",
                                            "conv6", "fc_super", "fc"]:
                                    param.requires_grad = True
                                else:
                                    param.requires_grad = False
                        else:
                            for name, param in model.named_parameters():
                                name = name.split(".")[0]
                                if name in ["conv1", "conv2", "conv3", "pre_fc_super", "fc_super", "conv4", "conv5",
                                            "conv6", "fc_super", "fc"]:
                                    param.requires_grad = False
                                else:
                                    param.requires_grad = True

                running_loss += loss.item()
                running_acc += torch.sum(class_preds == class_labels.data).item()
                running_superacc += torch.sum(superclass_preds == superclass_labels.data).item()
                running_attention_acc += torch.sum(attention_preds == class_labels.data).item()

            epoch_acc = running_acc / dataset_sizes[phase]
            epoch_superacc = running_superacc / dataset_sizes[phase]
            epoch_attention = running_attention_acc / dataset_sizes[phase]
            epoch_loss = running_loss / len(loader)

            print(f"{phase}: loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}, super accuracy {epoch_superacc:.4f}, attention accuracy {epoch_attention:.4f}")






