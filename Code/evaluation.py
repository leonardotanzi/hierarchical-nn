import numpy as np
import torch

from utils import sparser2coarser
from tree import node_to_weights

from anytree.util import commonancestors
from anytree.search import find
from anytree import PreOrderIter


def hierarchical_accuracy(predicted, actual, tree, all_leaves, device):

    predicted = torch.softmax(predicted, dim=1) + 1e6
    predicted = torch.argmax(predicted, dim=1)

    node_actual_name = np.asarray(all_leaves)[np.asarray(actual.cpu(), dtype=np.int64)]
    node_pred_name = np.asarray(all_leaves)[np.asarray(predicted.cpu(), dtype=np.int64)]

    h_acc = 0.0

    # read each couple of actual and predicted node, compute the ancestor, find the depth and compute metric
    for i, (n1, n2) in enumerate(zip(node_actual_name, node_pred_name)):
        node_actual = find(tree, lambda node: node.name == n1)
        node_pred = find(tree, lambda node: node.name == n2)
        ca = commonancestors(node_actual, node_pred)
        depth = ca[-1].depth
        h_acc += depth / tree.height if n1 != n2 else 1

    return (h_acc / i) * 100


def hierarchical_error(predicted, actual, tree, all_leaves, device):

    predicted = torch.softmax(predicted, dim=1) + 1e6
    predicted = torch.argmax(predicted, dim=1)

    node_actual_name = np.asarray(all_leaves)[np.asarray(actual.cpu(), dtype=np.int64)]
    node_pred_name = np.asarray(all_leaves)[np.asarray(predicted.cpu(), dtype=np.int64)]

    h_error = 0.0

    # read each couple of actual and predicted node, compute the ancestor, find the depth and compute metric
    for i, (n1, n2) in enumerate(zip(node_actual_name, node_pred_name)):
        node_actual = find(tree, lambda node: node.name == n1)
        node_pred = find(tree, lambda node: node.name == n2)

        ca = commonancestors(node_actual, node_pred)
        depth = ca[-1].depth
        h_error += 2**(-depth)

    return h_error / i


def accuracy_coarser_classes(predicted, actual, coarser_labels, n_superclass, device):
    batch = predicted.size(0)

    predicted = torch.softmax(predicted, dim=1) + 1e-6

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


def evaluate_regularization(model, tree, all_leaves):
    penalty = 0.0
    for node in PreOrderIter(tree):
        if node.name != "root":
            leaves_node = node.leaves
            leaves_parent = node.parent.leaves
            beta_vec = model.fc.weight.data

            weights_node = node_to_weights(all_leaves, leaves_node, beta_vec)
            weights_parent = node_to_weights(all_leaves, leaves_parent, beta_vec)
            penalty += ((len(node.leaves)) ** 2) * ((torch.norm(weights_node - weights_parent)) ** 2)