import numpy as np
import torch
import torch.nn.functional as F
from utils import sparser2coarser, class_to_index
from anytree import Node, RenderTree, LevelOrderGroupIter


def cross_entropy(predicted, actual, reduction):
    actual_onehot = F.one_hot(actual, num_classes=predicted.shape[1])
    loss = -torch.sum(actual_onehot * torch.log(predicted))
    return loss if reduction == "sum" else loss / float(predicted.shape[0])


def hierarchical_cc_3levels(predicted, actual, medium_labels, coarse_labels, n_medium_class, n_coarse_class, device):

    batch = predicted.size(0)

    predicted = torch.softmax(predicted, dim=1) + 1e-6

    loss = cross_entropy(predicted, actual, reduction="sum")
    loss_dict = {"loss_fine": loss.item()}

    # define an empty vector which contains 20 superclasses prediction for each samples
    # predicted_medium = torch.zeros(batch, n_medium_class, dtype=torch.float32, device=device)
    # for k in range(n_medium_class):
    #     # obtain the indexes of the superclass number k
    #     indexes = list(np.where(medium_labels == k))[0]
    #     # for each index, sum all the probability related to that superclass
    #     # for each line, at the position k, you sum all the classe related to superclass k, so for k=0
    #     # the classes are 0 to 4
    #     predicted_medium[:, k] += torch.sum(predicted[:, indexes], dim=1)
    # medium_labels = torch.tensor(medium_labels).type(torch.int64).to(device)
    # actual_medium = sparser2coarser(actual, medium_labels)
    # loss_medium = cross_entropy(predicted_medium, actual_medium, reduction="sum")
    # loss_dict["loss_medium"] = loss_medium.item()
    #
    # loss += loss_medium
    #
    # # define an empty vector which contains 20 superclasses prediction for each samples
    # predicted_coarse = torch.zeros(batch, n_coarse_class, dtype=torch.float32, device=device)
    #
    # for k in range(n_coarse_class):
    #     # obtain the indexes of the superclass number k
    #     indexes = list(np.where(coarse_labels == k))[0]
    #     # for each index, sum all the probability related to that superclass
    #     # for each line, at the position k, you sum all the classe related to superclass k, so for k=0
    #     # the classes are 0 to 4
    #     predicted_coarse[:, k] += torch.sum(predicted[:, indexes], dim=1)
    #
    # coarse_labels = torch.tensor(coarse_labels).type(torch.int64).to(device)
    # actual_coarse = sparser2coarser(actual, coarse_labels)
    # loss_coarse = cross_entropy(predicted_coarse, actual_coarse, reduction="sum")
    # loss_dict["loss_coarse"] = loss_coarse.item()

    # loss += loss_coarse

    loss_dict["loss_medium"] = 0.0
    loss_dict["loss_coarse"] = 0.0

    return loss, loss_dict


def hierarchical_cc_tree(predicted, actual, coarse_labels, tree, n_class, n_superclass, model, w0, device,
                    hierarchical_loss, regularization, sp_regularization, weight_decay):
    batch = predicted.size(0)

    predicted = torch.softmax(predicted, dim=1) + 1e-6
    loss = cross_entropy(predicted, actual, reduction="sum")

    loss_dict = {"loss_fine": loss.item()}

    if hierarchical_loss:
        # define an empty vector which contains 20 superclasses prediction for each samples
        predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device=device)

        for k in range(n_superclass):
            # obtain the indexes of the superclass number k
            indexes = list(np.where(coarse_labels == k))[0]
            # for each index, sum all the probability related to that superclass
            # for each line, at the position k, you sum all the classe related to superclass k, so for k=0
            # the classes are 0 to 4
            predicted_coarse[:, k] += torch.sum(predicted[:, indexes], dim=1)
            # this line is like the cycle below but more fast
            # for j in indexes:
            #     predicted_coarse[:, k] = predicted_coarse[:, k] + predicted[:, j]

        coarse_labels = torch.tensor(coarse_labels).type(torch.int64).to(device)
        actual_coarse = sparse2coarse(actual, coarse_labels)

        # loss_coarse = F.cross_entropy(predicted_coarse, torch.from_numpy(actual_coarse).type(torch.int64).to(device), reduction="sum")
        loss_coarse = cross_entropy(predicted_coarse, actual_coarse, reduction="sum")

        loss_dict["loss_coarse"] = loss_coarse.item()
        loss += loss_coarse

    else:
        loss_dict["loss_coarse"] = 0.0

    if regularization:

        penalty = 0
        for i, node in enumerate(LevelOrderGroupIter(tree)):
            # start computing not for root but for first layer
            if i > 0:
                # iniziare a ciclare sul primo livello (nel caso cifar, superclassi)
                for level_class in node:
                    # se sopra di loro c'è root non ho ancestor (quindi il secondo termine nell'equazione è =0
                    n_ancestors = 0 if i == 1 else 1
                    # prendo tutte le foglie del nodo (root avrà 100, una superclasse ne ha 5)
                    descendants = level_class.leaves
                    # PRIMO TERMINE
                    # se sono allultimo livello (quello delle classi, dove la heigth è zero,
                    # la formula è beta - mean(beta_parent), quindi devo prendere un solo vettore dai pesi
                    # come primo termine
                    if level_class.height == 0:
                        position = class_to_index(level_class.name)
                        beta_vec_node = model.fc.weight.data[position][None, :]
                    # se sono in un altro livello vado invece a prendere tutti i beta relativi alle leaf
                    else:
                        for j, classes_name in enumerate(descendants):
                            # recupero l'indice associato al nome della classe
                            position = class_to_index(classes_name.name)
                            # prendo il vettore tra i pesi relativo a quell'indice
                            beta_vec_node = model.fc.weight.data[position][None, :] if j == 0 else torch.cat((beta_vec_node, model.fc.weight.data[position][None, :]), 0)
                    # SECONDO TERMINE
                    # I have to do the same thing but this time with the leaves of the parent
                    if n_ancestors is not 0:
                        for k, superclasses_name in enumerate(level_class.ancestors[i-1].leaves):
                            position = class_to_index(superclasses_name.name)
                            beta_vec_parent = model.fc.weight.data[position][None, :] if k == 0 else torch.cat((beta_vec_parent, model.fc.weight.data[position][None, :]), 0)

                        # se n_ancestor è zero significa che il secondo termine non c'è, è il caso del primo livello
                        penalty += torch.linalg.norm(len(descendants) * (torch.mean(beta_vec_node, dim=0) - torch.mean(beta_vec_parent, dim=0)))
                    else:
                        penalty += torch.linalg.norm(len(descendants) * (torch.mean(beta_vec_node, dim=0)))

        print(f"Penalty:{penalty}")

        loss += weight_decay * penalty

        loss_dict["fine_penalty"] = fine_penalty.item()
        loss_dict["coarse_penalty"] = coarse_penalty.item()
        loss += weight_decay * (fine_penalty + coarse_penalty)

    else:
        loss_dict["fine_penalty"] = 0.0
        loss_dict["coarse_penalty"] = 0.0

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


def hierarchical_cc(predicted, actual, coarse_labels, n_class, n_superclass, model, w0, device,
                    hierarchical_loss, regularization, sp_regularization, weight_decay):
    batch = predicted.size(0)
    # compute the loss for fine classes
    # loss = F.cross_entropy(predicted, actual, reduction="sum")

    # predicted = F.log_softmax(predicted, dim=1)
    # loss = F.nll_loss(predicted, actual, reduction="sum")

    predicted = torch.softmax(predicted, dim=1) + 1e-6
    loss = cross_entropy(predicted, actual, reduction="sum")

    loss_dict = {"loss_fine": loss.item()}

    if hierarchical_loss:
        # define an empty vector which contains 20 superclasses prediction for each samples
        predicted_coarse = torch.zeros(batch, n_superclass, dtype=torch.float32, device=device)

        for k in range(n_superclass):
            # obtain the indexes of the superclass number k
            indexes = list(np.where(coarse_labels == k))[0]
            # for each index, sum all the probability related to that superclass
            # for each line, at the position k, you sum all the classe related to superclass k, so for k=0
            # the classes are 0 to 4
            predicted_coarse[:, k] += torch.sum(predicted[:, indexes], dim=1)
            # this line is like the cycle below but more fast
            # for j in indexes:
            #     predicted_coarse[:, k] = predicted_coarse[:, k] + predicted[:, j]

        coarse_labels = torch.tensor(coarse_labels).type(torch.int64).to(device)
        actual_coarse = sparse2coarse(actual, coarse_labels)

        # loss_coarse = F.cross_entropy(predicted_coarse, torch.from_numpy(actual_coarse).type(torch.int64).to(device), reduction="sum")
        loss_coarse = cross_entropy(predicted_coarse, actual_coarse, reduction="sum")

        loss_dict["loss_coarse"] = loss_coarse.item()
        loss += loss_coarse

    else:
        loss_dict["loss_coarse"] = 0.0

    if regularization:
        coarse_penalty = 0.0
        # fine_penalty = 0.0
        mean_betas = []
        for i in range(n_superclass):
            coarse_penalty += (torch.linalg.norm(
                torch.sum(model.fc.weight[i * n_class:i * n_class + n_class], dim=0))) ** 2
            mean_betas.append(
                1 / n_class * torch.sum(model.fc.weight[i * n_class:i * n_class + n_class], dim=0).repeat(n_class, 1))
        fine_penalty = torch.sum(
            torch.linalg.norm(model.fc.weight - torch.cat(mean_betas, dim=0).view(n_class * n_superclass, 512),
                              dim=0) ** 2)

        # faster than
        # for i in range(n_class * n_superclass):
        #     sc_index = i//5
        #     fine_penalty += (torch.linalg.norm(model.fc.weight[i] - 1 / n_class * torch.sum(model.fc.weight[sc_index * n_class:sc_index * n_class + n_class], dim=0))) ** 2

        loss_dict["fine_penalty"] = fine_penalty.item()
        loss_dict["coarse_penalty"] = coarse_penalty.item()
        loss += weight_decay * (fine_penalty + coarse_penalty)

    else:
        loss_dict["fine_penalty"] = 0.0
        loss_dict["coarse_penalty"] = 0.0

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


def hierarchical_cc_bones(predicted, actual, fine_classes, medium_classes, coarse_classes, model, device,
                          hierarchical_loss, regularization, sp_regularization, weight_decay):
    batch = predicted.size(0)

    mapping_medium = torch.tensor([0, 0, 0, 1, 1, 1, 2], dtype=torch.int64)
    mapping_coarse = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.int64)

    predicted = torch.softmax(predicted, dim=1) + 1e-6
    loss = cross_entropy(predicted, actual, reduction="sum")

    loss_dict = {"loss_fine": loss.item()}

    if hierarchical_loss:
        # define an empty vector which contains 20 superclasses prediction for each samples
        predicted_medium = torch.zeros(batch, len(medium_classes), dtype=torch.float32, device=device)
        predicted_coarse = torch.zeros(batch, len(coarse_classes), dtype=torch.float32, device=device)

        predicted_medium[:, 0] = torch.sum(predicted[:, 0:3], dim=1)
        predicted_medium[:, 1] = torch.sum(predicted[:, 3:6], dim=1)
        predicted_medium[:, 2] = predicted[:, 6]

        predicted_coarse[:, 0] = torch.sum(predicted[:, 0:6], dim=1)
        predicted_coarse[:, 1] = predicted[:, 6]

        actual_medium = mapping_medium[actual]
        actual_coarse = mapping_coarse[actual]

        # loss_coarse = F.cross_entropy(predicted_coarse, torch.from_numpy(actual_coarse).type(torch.int64).to(device), reduction="sum")
        loss_medium = cross_entropy(predicted_medium, actual_medium.to(device), reduction="sum")
        loss_coarse = cross_entropy(predicted_coarse, actual_coarse.to(device), reduction="sum")

        loss_dict["loss_medium"] = loss_medium.item()
        loss_dict["loss_coarse"] = loss_coarse.item()
        loss += loss_medium + loss_coarse

    else:
        loss_dict["loss_medium"] = 0.0
        loss_dict["loss_coarse"] = 0.0

    if regularization:
        coarse_penalty = 0.0
        # fine_penalty = 0.0
        mean_betas = []
        for i in range(n_superclass):
            coarse_penalty += (torch.linalg.norm(
                torch.sum(model.fc.weight[i * n_class:i * n_class + n_class], dim=0))) ** 2
            mean_betas.append(
                1 / n_class * torch.sum(model.fc.weight[i * n_class:i * n_class + n_class], dim=0).repeat(n_class, 1))
        fine_penalty = torch.sum(
            torch.linalg.norm(model.fc.weight - torch.cat(mean_betas, dim=0).view(n_class * n_superclass, 512),
                              dim=0) ** 2)

        # faster than
        # for i in range(n_class * n_superclass):
        #     sc_index = i//5
        #     fine_penalty += (torch.linalg.norm(model.fc.weight[i] - 1 / n_class * torch.sum(model.fc.weight[sc_index * n_class:sc_index * n_class + n_class], dim=0))) ** 2

        loss_dict["fine_penalty"] = fine_penalty.item()
        loss_dict["coarse_penalty"] = coarse_penalty.item()
        loss += weight_decay * (fine_penalty + coarse_penalty)

    else:
        loss_dict["fine_penalty"] = 0.0
        loss_dict["coarse_penalty"] = 0.0

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
