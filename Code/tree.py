from anytree import Node, RenderTree
from anytree.search import find
import torch
import numpy as np
import plotly.express as px


def node_to_weights(classes, node, beta):
    w = []
    for n in node:
        index = classes.index(n.name)
        w.append(beta[index])
    w = torch.cat(w, 0).view(-1, beta.shape[1])
    return torch.mean(w, 0)


def count_symbol(line):
    L = 0
    for i in line:
        if i == "-":
            L += 1
        else:
            break
    return L


def get_tree_from_file(file_tree):

    file = open(file_tree, "r")
    all_lines = file.readlines()
    all_lines = [line[:-1] for line in all_lines]
    root = Node("root")

    R = root
    L_before = 0

    for line in all_lines[1:]:
        # count the number of "-" at the beginning
        L_actual = count_symbol(line)

        # if it's bigger, mean we are going into a next level
        if L_actual > L_before:
            # update the value of L
            L_before = L_actual
            # create a node with the parent
            node = Node(f"{line[L_actual:]}", parent=R)
            # update the parent
            R = node
        # if it's in the same level
        elif L_actual == L_before:
            # i need the parent, I am not coing deeper in the hierarchy
            R = node.parent
            node = Node(f"{line[L_actual:]}", parent=R)
            # update the node
            R = node
        else:
            # if we have to return back, firstly assign the parent as root
            R = node.parent
            # then we add one parent for each couple o "-"
            for i in range(((L_before - L_actual) // 2)):
                R = R.parent
            node = Node(f"{line[L_actual:]}", parent=R)
            # update the node
            R = node
            L_before = L_actual

    print(RenderTree(root))
    print("-" * 100)

    return root


def write_file_tree_fgvc():
    ds = {}
    ds_man = {}
    ds_fam = {}

    with open("..//..//Dataset//fgvc-aircraft-2013b//data//images_manufacturer_trainval.txt", "r") as file:

        for line in file.readlines():
            line = line[:-1]
            token0 = line.split(" ")[0]
            token1 = line[len(token0) + 1:]
            ds[token0] = [token1]

    with open("..//..//Dataset//fgvc-aircraft-2013b//data//images_family_trainval.txt", "r") as file:

        for line in file.readlines():
            line = line[:-1]
            token0 = line.split(" ")[0]
            token1 = line[len(token0) + 1:]
            ds[token0].append(token1)

    with open("..//..//Dataset//fgvc-aircraft-2013b//data//images_variant_trainval.txt", "r") as file:

        for line in file.readlines():
            line = line[:-1]
            token0 = line.split(" ")[0]
            token1 = line[len(token0) + 1:]
            ds[token0].append(token1)

    print(ds)

    with open("FGVCtree.txt", "w") as file:
        written_man = []
        for values in ds.values():
            if values[0] not in written_man:
                file.write(values[0] + "\n")
                written_fam = []
                written_mod = []
                written_man.append(values[0])

            if values[1] not in written_fam:
                file.write("--" + values[1] + "\n")
                written_mod = []
                written_fam.append(values[1])

            if values[2] not in written_mod:
                file.write("----" + values[2] + "\n")
                written_mod.append(values[2])


def get_tree_FashionMNIST():

    superclass_dict = {'Up': {'Short': ['T-shirt/top', 'Pullover', 'Shirt'], 'Long': ['Dress', 'Coat']},
                       'Trouser': [], 'Bag': [], 'Shoes': ['Sandal', 'Sneaker', 'Ankle boot']}
    root = Node("root")
    for key, value in superclass_dict.items():
        parent = Node(f"{key}", parent=root)
        if type(value) is dict:
            for key_next, value_next in value.items():
                node = Node(f"{key_next}", parent=parent)
                for classes in value_next:
                    node2 = Node(f"{classes}", parent=node)
        else:
            for classes in value:
                node = Node(f"{classes}", parent=parent)

    print(RenderTree(root))
    return root


def get_tree_limited_CIFAR():

    # superclass_dict = {'sea animal': {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    #                                 'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']},
    #                    'land animal': {'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    #                                    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    #                                    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    #                                    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    #                                    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']},
    #
    #                    'insect and invertebrates': {'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    #                                                 'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm']},
    #
    #                    'flora': {'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    #                              'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    #                              'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']},
    #
    #                    'object': {'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    #                               'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    #                               'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe']},
    #                    'outdoor scenes': {'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    #                                       'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea']},
    #                    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    #                    'vehicles': {'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    #                                 'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}}

    superclass_dict = {'sea animal': {'aquatic mammals': ['beaver', 'dolphin'],
                                      'fish': ['aquarium_fish', 'flatfish']},
                       'people': ['baby', 'boy', 'girl', 'man'],
                       'flora': {'flowers': ['orchid', 'poppy', 'tulip'],
                                 'fruit and vegetables': ['apple', 'mushroom']
                                 }}

    # # # assegno la somma delle leaves ai nodi maggiori e i valori singoli ai leaf, mantendendo la batch size
    root = Node("root")
    for key, value in superclass_dict.items():
        parent = Node(f"{key}", parent=root)
        if type(value) is dict:
            for key_next, value_next in value.items():
                node = Node(f"{key_next}", parent=parent)
                for classes in value_next:
                    node2 = Node(f"{classes}", parent=node)
        else:
            for classes in value:
                node = Node(f"{classes}", parent=parent)

    print(RenderTree(root))
    return root


def get_all_labels_downtop(tree):

    all_labels_node = []
    # to convert the fine labels to any other level, read each node, count the leaves and add one integer for each
    # leaves in the node

    all_leaves_node = [leaf for leaf in tree.leaves]
    for i in range(tree.height - 1):
        labels = []
        for leaf in all_leaves_node:
            # check level ito check if all the leaf are equal for this level. to explain, this was made to solve the problem of
            # root
            # --shoes
            # ----sandal
            # ------sandal1
            # ------sandal2
            # ----sneaker
            # if i do not do this, at first the label will be [sandal1, sandal2, sneaker] and then [sandal, shoes]
            # because i will put in the list the ancestor of sneaker. in this way 'sneaker' wait sandal before going to shoes
            check_level = True
            for c in leaf.ancestors[-1].children:
                if c not in all_leaves_node:
                    check_level = False
                    break
            if len(leaf.ancestors) != 1 and check_level:
                labels.append(leaf.ancestors[-1])
            else:
                labels.append(leaf)
        all_leaves_node = labels
        all_labels_node.append(labels)

    all_labels = []
    for label in all_labels_node:
        i = 0
        number = []
        for counter, current_label in enumerate(label):
            if counter == 0:
                number.append(i)
                prev_label = current_label
            else:
                if current_label.name != prev_label.name:
                    i += 1
                number.append(i)
            prev_label = current_label
        all_labels.append(number)
    all_labels.reverse()
    return all_labels


def get_all_labels_topdown(tree):

    all_labels = []
    children = list(tree.children)
    for i in range(tree.height - 1):
        labels = []
        next_children = []
        for j, child in enumerate(children):
            for _ in range(len(child.leaves)):
                labels.append(j)
        for child in children:
            if len(child.children) == 0:
                next_children.append(child.leaves)
            else:
                next_children.append(child.children)
        children = [child for sublist in next_children for child in sublist]
        all_labels.append(labels)
    return all_labels


def return_matrixes_downtop(tree, plot=False):
    matrixes = []
    # read each level and put all the classes name in separate lists
    all_leaves = [leaf.name for leaf in tree.leaves]

    all_labels_node = []
    # read the superclasses for each layer and store in a list of list
    all_leaves_node = [leaf for leaf in tree.leaves]
    for i in range(tree.height - 1):
        labels = []
        for leaf in all_leaves_node:
            check_level = True
            for c in leaf.ancestors[-1].children:
                if c not in all_leaves_node:
                    check_level = False
                    break
            if len(leaf.ancestors) != 1 and check_level:
                to_append = leaf.ancestors[-1]
            else:
                to_append = leaf
            if to_append not in labels:
                labels.append(to_append)

        all_leaves_node = labels
        all_labels_node.append(labels)

    # reverse the list and substitute each node with its actual name
    all_labels_node.reverse()
    all_labels_node = [[node.name for node in all_labels] for all_labels in all_labels_node]

    # build one matrix for each layer that is not the last (we dont a matrix for the last)
    for node_layer in all_labels_node:

        # the size of the matrix is this because we have one entry for each of the superclass of the level we are
        # considering, compared with each one of the leaf class (all_nodes[-1])
        matrix = np.zeros((len(all_leaves), len(node_layer)))

        for i, node_name in enumerate(node_layer):
            # we find the actual node of the tree in order to extract the leaves
            actual_node = find(tree, lambda node: node.name == node_name)
            leaves = actual_node.leaves
            # for each leaf i extract the index of the corresponding leaf and i set to 1 the entry in the matrix
            for leaf in leaves:
                index = all_leaves.index(leaf.name)
                matrix[index][i] = 1
        matrixes.append(matrix)

        if plot:
            fig = px.imshow(matrix, text_auto=True, aspect="auto", x=node_layer, y=all_leaves, width=2500 // 6, height=2500 // 6)
            fig.update_xaxes(side="top")
            fig.show()

    return matrixes


def return_matrixes_topdown(tree, plot=False):
    matrixes = []
    # read each level and put all the classes name in separate lists
    all_leaves = [leaf.name for leaf in tree.leaves]

    all_labels_node = []
    # read the superclasses for each layer and store in a list of list
    all_leaves_node = [leaf for leaf in tree.leaves]
    children = list(tree.children)
    for i in range(tree.height - 1):
        all_labels_node.append(children)
        labels = []
        next_children = []
        for child in children:
            if len(child.children) == 0:
                next_children.append(child.leaves)
            else:
                next_children.append(child.children)
        children = [child for sublist in next_children for child in sublist]

    # reverse the list and substitute each node with its actual name
    # all_labels_node.reverse()
    all_labels_node = [[node.name for node in all_labels] for all_labels in all_labels_node]

    # build one matrix for each layer that is not the last (we dont a matrix for the last)
    for node_layer in all_labels_node:

        # the size of the matrix is this because we have one entry for each of the superclass of the level we are
        # considering, compared with each one of the leaf class (all_nodes[-1])
        matrix = np.zeros((len(all_leaves), len(node_layer)))

        for i, node_name in enumerate(node_layer):
            # we find the actual node of the tree in order to extract the leaves
            actual_node = find(tree, lambda node: node.name == node_name)
            leaves = actual_node.leaves
            # for each leaf i extract the index of the corresponding leaf and i set to 1 the entry in the matrix
            for leaf in leaves:
                index = all_leaves.index(leaf.name)
                matrix[index][i] = 1
        matrixes.append(matrix)

        if plot:
            fig = px.imshow(matrix, text_auto=True, aspect="auto", x=node_layer, y=all_leaves, width=2500 // 6, height=2500 // 6)
            fig.update_xaxes(side="top")
            fig.show()

    return matrixes


if __name__ == "__main__":
    tree = get_tree_from_file("..\\..\\Dataset\\tree_paper.txt")
    return_matrixes_topdown(tree, plot=False)
