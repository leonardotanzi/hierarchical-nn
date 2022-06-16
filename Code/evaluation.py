import numpy as np
import torch
from utils import sparser2coarser


def accuracy_superclasses(predicted, actual, coarse_labels, n_superclass, device):
    batch = predicted.size(0)
    predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device="cuda:0")

    for k in range(n_superclass):
        # obtain the indexes of the superclass number k
        indexes = list(np.where(coarse_labels == k))[0]
        predicted_coarse[:, k] += torch.sum(predicted[:, indexes], dim=1)

    coarse_labels = torch.tensor(coarse_labels).type(torch.int64).to(device)
    actual_coarse = sparser2coarser(actual, coarse_labels)

    # actual_coarse = sparse2coarse(actual.cpu().detach().numpy(), coarse_labels)
    predicted_coarse = torch.argmax(predicted_coarse, dim=1)
    running_corrects = torch.sum(predicted_coarse == actual_coarse)

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
