import numpy as np
import torch
import torch.nn.functional as F
from utils import sparser2coarser, class_to_index
from anytree import Node, RenderTree, LevelOrderGroupIter, PreOrderIter
from tree import return_matrixes_downtop, node_to_weights


def cross_entropy(predicted, actual, reduction):
    actual_onehot = F.one_hot(actual, num_classes=predicted.shape[1])
    loss = -torch.sum(actual_onehot * torch.log(predicted))
    return loss if reduction == "sum" else loss / float(predicted.shape[0])


def hierarchical_cc_treebased(predicted, actual, tree, lens, all_labels, all_leaves, model, w0, device, hierarchical_loss,
                              regularization, sp_regularization, weight_decay, matrixes, architecture, multigpu=False):

    batch = predicted.size(0)

    predicted = torch.softmax(predicted, dim=1) + 1e-12
    loss = cross_entropy(predicted, actual, reduction="sum")

    loss_dict = {"loss_fine": loss.item()}

    if hierarchical_loss:
        loss_hierarchical = 0.0
        # start from the root i compute all the other losses
        for i, labels in enumerate(all_labels):
            labels_coarser = sparser2coarser(actual, torch.tensor(labels, requires_grad=False).type(torch.int64).to(device))
            predicted_coarser = torch.matmul(predicted, torch.tensor(matrixes[i], requires_grad=False).type(torch.float32).to(device))
            loss_coarser = cross_entropy(predicted_coarser, labels_coarser, reduction="sum")
            loss_dict[f"loss_{i}"] = loss_coarser.item()
            loss_hierarchical += loss_coarser
        loss += loss_hierarchical

    else:
        for i, labels in enumerate(all_labels):
            loss_dict[f"loss_{i}"] = 0.0

    if regularization:
        penalty = 0.0
        for node in PreOrderIter(tree):
            if node.name != "root":
                leaves_node = node.leaves
                leaves_parent = node.parent.leaves
                if multigpu:
                    beta_vec = model.module.fc.weight.data
                else:
                    if architecture == "vit":
                        beta_vec = model.classifier.weight.data
                    else:
                        beta_vec = model.fc.weight.data
                weights_node = node_to_weights(all_leaves, leaves_node, beta_vec)
                weights_parent = node_to_weights(all_leaves, leaves_parent, beta_vec)
                penalty += ((len(node.leaves))**2)*((torch.norm(weights_node - weights_parent))**2)

        loss += weight_decay * penalty
    if sp_regularization:
        w = []
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'weight' in name and 'fc' not in name:
                w.append(W.view(-1))
        sp_reg = (torch.linalg.norm(torch.cat(w) - w0)) ** 2
        loss_dict["sp_reg"] = sp_reg.item()
        loss += weight_decay * sp_reg
    else:
        loss_dict["sp_reg"] = 0.0

    return loss, loss_dict

