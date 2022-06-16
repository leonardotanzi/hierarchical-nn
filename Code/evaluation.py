import numpy as np
import torch
from utils import sparser2coarser


def accuracy_coarser_classes(predicted, actual, coarser_labels, n_superclass, device):
    batch = predicted.size(0)
    predicted_coarser = torch.zeros(batch, n_superclass, dtype=torch.float32, device="cuda:0")

    for k in range(n_superclass):
        # obtain the indexes of the superclass number k
        indexes = list(np.where(coarser_labels == k))[0]
        predicted_coarser[:, k] += torch.sum(predicted[:, indexes], dim=1)

    coarser_labels = torch.tensor(coarser_labels).type(torch.int64).to(device)
    actual_coarser = sparser2coarser(actual, coarser_labels)

    predicted_coarser = torch.argmax(predicted_coarser, dim=1)
    running_corrects = torch.sum(predicted_coarser == actual_coarser)

    return running_corrects


def evaluate_regularization(model, n_class=5, n_superclass=20):
    coarse_penalty = 0.0
    fine_penalty = 0.0
    for i in range(n_superclass):
        coarse_penalty += (torch.linalg.norm(
            torch.sum(model.fc.weight.data[i * n_class:i * n_class + n_class], dim=0))) ** 2
    for i in range(n_class * n_superclass):
        sc_index = 1 // 5
        fine_penalty += (torch.linalg.norm(model.fc.weight.data[i] - 1 / n_class * torch.sum(
            model.fc.weight.data[sc_index * n_class:sc_index * n_class + n_class]))) ** 2

    print(coarse_penalty)
    print(fine_penalty)
