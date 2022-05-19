from torchsummary import summary
from torch.optim import SGD
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
import torch.nn.functional as F
from utils import ClassSpecificImageFolderNotAlphabetic, train_val_dataset, exclude_classes, get_classes, sparse2coarse
import numpy as np
import torch

def double_loss(predicted, actual, coarse_labels):

    predict_classes = predicted[0]
    predict_superclasses = predicted[1]

    actual_coarse = torch.from_numpy(sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)).type(torch.int64).to("cuda:0")

    loss_classes = F.cross_entropy(predict_classes, actual, reduction="sum")
    loss_superclasses = F.cross_entropy(predict_superclasses, actual_coarse, reduction="sum")
    return loss_classes + loss_superclasses


class ConvNetIntermediate(nn.Module):
    def __init__(self, num_superclasses, num_classes):
        super(ConvNetIntermediate, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same")
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_h = nn.Linear(4096, 2048)
        self.fc_cl1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc_cl2 = nn.Linear(2048, num_classes)
        self.fc_scl1 = nn.Linear(128 * 8 * 8, 4096)
        self.fc_scl2 = nn.Linear(2048, num_superclasses)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #first branch for prediction
        x_intermediate = x.view(-1, 128 * 8 * 8)
        x_intermediate = F.relu(self.fc_scl1(x_intermediate))
        x_intermediate = F.relu(self.fc_h(x_intermediate))
        x_superclasses = self.fc_scl2(x_intermediate)

        #reshape the last layer before prediction from 2048 to 128*4*4 and then upsample to match the size
        x_intermediate = x_intermediate.view((-1, 128, 4, 4))
        x_intermediate = nn.Upsample(scale_factor=2)(x_intermediate)

        # second branch
        x = F.relu(self.conv3(x + x_intermediate))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc_cl1(x))
        x = F.relu(self.fc_h(x))
        x_classes = self.fc_cl2(x)

        return x_classes, x_superclasses


if __name__ == "__main__":

    device = "cuda:0"

    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dir = "../../../cifar/train//"
    model_name = "..//..//cnn_hierarchical_3classes_intermediate_sum.pth"

    batch_size = 128
    learning_rate = 0.001
    image_size = 32
    num_epochs = 1000

    classes_name = get_classes()
    superclasses = ["flowers", "fruit and vegetables", "trees"]

    excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)

    classes_name.append(excluded)

    train_dataset = ClassSpecificImageFolderNotAlphabetic(train_dir, all_dropped_classes=classes_name,
                                                          transform=transform)

    dataset = train_val_dataset(train_dataset, val_split=0.15)
    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}
    class_names = dataset['train'].dataset.classes

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    model = ConvNetIntermediate(num_classes=15, num_superclasses=3).to(device)
    # print(summary(model, (3, 32, 32)))

    optimizer = SGD(model.parameters(), lr=0.001)

    n_total_steps_train = len(train_loader)
    n_total_steps_val = len(val_loader)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs - 1))
        print(f"Best acc: {best_acc:.4f}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                loader = train_loader
                n_total_steps = n_total_steps_train
            else:
                model.eval()  # Set model to evaluate mode
                loader = val_loader
                n_total_steps = n_total_steps_val

            # vengono set a zero sia per train che per valid
            running_loss = 0.0
            running_corrects = 0

            for i, (images, labels) in enumerate(loader):

                images = images.to(device)
                labels = labels.to(device)

                # forward
                # track history only if train
                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(images)
                    _, preds = torch.max(outputs[0], 1)
                    loss = double_loss(outputs, labels, np.asarray(coarse_labels))
                    # loss = classic_cc(outputs, labels)
                    # backward + optimize if training
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print("Step {}/{}, {} Loss: {:.4f} Acc: {:.4f}".format(i + 1, n_total_steps, phase, epoch_loss,
                                                                       epoch_acc))

                if phase == "val" and (i + 1) % n_total_steps == 0 and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), model_name)
                    print("New best accuracy {}, saving best model".format(best_acc))

    print("Finished Training")