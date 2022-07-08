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
                              regularization, sp_regularization, weight_decay, matrixes, multigpu=False):

    batch = predicted.size(0)

    predicted = torch.softmax(predicted, dim=1) + 1e-6
    loss = 0.0 #cross_entropy(predicted, actual, reduction="sum")

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
                    beta_vec = model.fc.weight.data
                weights_node = node_to_weights(all_leaves, leaves_node, beta_vec)
                weights_parent = node_to_weights(all_leaves, leaves_parent, beta_vec)
                penalty += ((len(node.leaves))**2)*((torch.norm(weights_node - weights_parent))**2)

        loss += weight_decay * penalty
    # if regularization:
    #
    #     penalty = 0
    #     for i, node in enumerate(LevelOrderGroupIter(tree)):
    #         # start computing not for root but for first layer
    #         if i > 0:
    #             # iniziare a ciclare sul primo livello (nel caso cifar, superclassi)
    #             for level_class in node:
    #                 # se sopra di loro c'è root non ho ancestor (quindi il secondo termine nell'equazione è =0
    #                 n_ancestors = 0 if i == 1 else 1
    #                 # prendo tutte le foglie del nodo (root avrà 100, una superclasse ne ha 5)
    #                 descendants = level_class.leaves
    #                 # PRIMO TERMINE
    #                 # se sono allultimo livello (quello delle classi, dove la heigth è zero,
    #                 # la formula è beta - mean(beta_parent), quindi devo prendere un solo vettore dai pesi
    #                 # come primo termine
    #                 if level_class.height == 0:
    #                     position = class_to_index(level_class.name)
    #                     beta_vec_node = model.fc.weight.data[position][None, :]
    #                 # se sono in un altro livello vado invece a prendere tutti i beta relativi alle leaf
    #                 else:
    #                     for j, classes_name in enumerate(descendants):
    #                         # recupero l'indice associato al nome della classe
    #                         position = class_to_index(classes_name.name)
    #                         # prendo il vettore tra i pesi relativo a quell'indice
    #                         beta_vec_node = model.fc.weight.data[position][None, :] if j == 0 else torch.cat((beta_vec_node, model.fc.weight.data[position][None, :]), 0)
    #                 # SECONDO TERMINE
    #                 # I have to do the same thing but this time with the leaves of the parent
    #                 if n_ancestors is not 0:
    #                     for k, superclasses_name in enumerate(level_class.ancestors[i-1].leaves):
    #                         position = class_to_index(superclasses_name.name)
    #                         beta_vec_parent = model.fc.weight.data[position][None, :] if k == 0 else torch.cat((beta_vec_parent, model.fc.weight.data[position][None, :]), 0)
    #
    #                     # se n_ancestor è zero significa che il secondo termine non c'è, è il caso del primo livello
    #                     penalty += torch.linalg.norm(len(descendants) * (torch.mean(beta_vec_node, dim=0) - torch.mean(beta_vec_parent, dim=0)))
    #                 else:
    #                     penalty += torch.linalg.norm(len(descendants) * (torch.mean(beta_vec_node, dim=0)))
    #
    #     print(f"Penalty:{penalty}")
    #
    #     loss += weight_decay * penalty
    #
    #     loss_dict["fine_penalty"] = fine_penalty.item()
    #     loss_dict["coarse_penalty"] = coarse_penalty.item()
    #     loss += weight_decay * (fine_penalty + coarse_penalty)
    #
    # else:
    #     loss_dict["fine_penalty"] = 0.0
    #     loss_dict["coarse_penalty"] = 0.0

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

