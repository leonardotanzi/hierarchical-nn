import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import ClassSpecificImageFolderNotAlphabetic, imshow, train_val_dataset, sparse2coarse, exclude_classes, \
    get_classes, get_superclasses, accuracy_superclasses
from EfficientNetV2 import effnetv2_l
import torch.nn.functional as F
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
import numpy as np


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

#
#
# if __name__ == "__main__":
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), Resize(size=(224, 224))])
#
#     train_dir = "..//..//cifar//train//"
#     test_dir = "..//..//cifar//test//"
#
#     image_size = 32
#
#     num_epochs = 200
#     batch_size = 4
#     learning_rate = 0.001
#     early_stopping = 200
#
#     hierarchical_loss = False
#     weight_decay1 = 0.1
#     weight_decay2 = 0.1
#     all_superclasses = True
#     less_samples = True
#     reduction_factor = 1
#
#     model_name = "..//..//effnet2SAM-hloss-1on{}-all.pth".format(reduction_factor) if hierarchical_loss else "..//..//cvt-1on{}-all.pth".format(reduction_factor)
#
#     classes_name = get_classes()
#
#     if not all_superclasses:
#         # read superclasses, you can manually select some or get all with get_superclasses()
#         superclasses = ["flowers", "fruit and vegetables", "trees"]
#     else:
#         superclasses = get_superclasses()
#
#     # given the list of superclasses, returns the class to exclude and the coarse label
#     excluded, coarse_labels = exclude_classes(superclasses_names=superclasses)
#
#     classes_name.append(excluded)
#
#     # take as input a list of list with the first element being the classes_name and the second the classes to exclude
#     train_dataset = ClassSpecificImageFolderNotAlphabetic(train_dir, all_dropped_classes=classes_name, transform=transform)
#     test_dataset = ClassSpecificImageFolderNotAlphabetic(test_dir, all_dropped_classes=classes_name, transform=transform)
#
#     if less_samples:
#         evens = list(range(0, len(train_dataset), reduction_factor))
#         train_dataset = torch.utils.data.Subset(train_dataset, evens)
#
#     dataset = train_val_dataset(train_dataset, val_split=0.15)
#
#     train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)
#
#     dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}
#     if less_samples:
#         class_names = dataset['train'].dataset.dataset.classes
#     else:
#         class_names = dataset['train'].dataset.classes
#
#     num_class, num_superclass = len(class_names), len(superclasses)
#
#     print(class_names)
#
#     # Network
#     model = effnetv2_l(num_classes=num_class)
#     model.to(device)
#
#     optimizer = torch.optim.SGD
#     sam_optimizer = SAM(model.parameters(), optimizer, lr=0.001, momentum=0.9)
#
#     criterion = nn.CrossEntropyLoss(reduction="mean")
#
#     n_total_steps_train = len(train_loader)
#     n_total_steps_val = len(val_loader)
#
#     best_acc = 0.0
#     associated_sup_acc = 0.0
#     platoon = 0
#     stop = False
#
#     for epoch in range(num_epochs):
#
#         if stop:
#             break
#
#         print("Epoch {}/{}".format(epoch + 1, num_epochs))
#         print(f"Best acc: {best_acc:.4f}, associate best superclass acc: {associated_sup_acc:.4f}")
#         print("-" * 30)
#
#         # Each epoch has a training and validation phase
#         for phase in ["train", "val"]:
#             if phase == "train":
#                 model.train()  # Set model to training mode
#                 loader = train_loader
#                 n_total_steps = n_total_steps_train
#             else:
#                 model.eval()  # Set model to evaluate mode
#                 loader = val_loader
#                 n_total_steps = n_total_steps_val
#
#             # vengono set a zero sia per train che per valid
#             running_loss = 0.0
#             running_corrects = 0
#
#             for i, (images, labels) in enumerate(loader):
#
#                 images = images.to(device)
#                 labels = labels.to(device)
#
#                 # forward
#                 # track history only if train
#                 with torch.set_grad_enabled(phase == "train"):
#
#                     enable_running_stats(model)
#                     outputs = model(images)
#                     _, preds = torch.max(outputs, 1)
#
#                     if hierarchical_loss:
#                         loss = hierarchical_cc(outputs, labels, np.asarray(coarse_labels), int(num_class/num_superclass), num_superclass,
#                                                model, weight_decay1=weight_decay1, weight_decay2=weight_decay2)
#                     else:
#                         loss = F.cross_entropy(outputs, labels, reduction="mean")
#
#                     # backward + optimize if training
#                     if phase == "train":
#
#                         loss.backward()
#                         sam_optimizer.first_step(zero_grad=True)
#                         # second forward-backward pass
#                         disable_running_stats(model)
#                         F.cross_entropy(model(images), labels, reduction="mean").backward()  # make sure to do a full forward pass
#                         sam_optimizer.second_step(zero_grad=True)
#
#                     running_loss += loss.item() * images.size(0)
#                     running_corrects += torch.sum(preds == labels.data)
#
#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#                 acc_super = accuracy_superclasses(outputs, labels, np.asarray(coarse_labels), len(superclasses))
#
#                 print("Step {}/{}, {} Loss: {:.4f} Acc: {:.4f} Acc Super: {:.4f}".format(i + 1, n_total_steps, phase, epoch_loss, epoch_acc, acc_super))
#
#                 if phase == "val" and (i+1) % n_total_steps == 0 and epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     associated_sup_acc = acc_super
#                     platoon = 0
#                     torch.save(model.state_dict(), model_name)
#                     print("New best accuracy {:.4f}, superclass accuracy {:.4f}, saving best model".format(best_acc, acc_super))
#
#                 if phase == "val" and (i+1) % n_total_steps == 0 and epoch_acc < best_acc:
#                     platoon += 1
#                     print("{} epochs without improvement".format(platoon))
#                     if platoon == early_stopping:
#                         print("Network didn't improve after {} epochs, early stopping".format(early_stopping))
#                         stop = True
#
#     print("Finished Training")
#
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     y_pred = []
#     y_true = []
#
#     # iterate over test data
#     for inputs, labels in test_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         output = model(inputs).logits  # Feed Network
#
#         output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#         y_pred.extend(output)  # Save Prediction
#
#         labels = labels.data.cpu().numpy()
#         y_true.extend(labels)  # Save Truth
#
#     # Build confusion matrix
#     cf_matrix = confusion_matrix(y_true, y_pred)
#     print(cf_matrix)