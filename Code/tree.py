from anytree import Node, RenderTree, LevelOrderGroupIter, NodeMixin, LevelOrderIter
from anytree.search import find
import torch
import numpy as np
import sys
import plotly.express as px


class MyNode(Node):  # Add Node feature
    def __init__(self, name, prob_value, parent=None, children=None):
        super(MyNode, self).__init__(name, parent=None, children=None)
        self.name = name
        self.parent = parent
        self.loss_value = prob_value
        if children:  # set children only if given
            self.children = children


def get_tree_CIFAR():

    # superclass_dict = {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    #                    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    #                    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    #                    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    #                    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    #                    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    #                    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    #                    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    #                    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    #                    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    #                    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    #                    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    #                    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    #                    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    #                    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    #                    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    #                    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    #                    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    #                    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    #                    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}

    superclass_dict = {'sea animal': {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                                    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']},
                       'land animal': {'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                                       'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                                       'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                                       'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                                       'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']},

                       'insect and invertebrates': {'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                                                    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm']},

                       'flora': {'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                                 'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                                 'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']},

                       'object': {'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
                                  'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                                  'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe']},
                       'outdoor scenes': {'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                                          'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea']},
                       'people': ['baby', 'boy', 'girl', 'man', 'woman'],
                       'vehicles': {'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                                    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}}

    # superclass_dict = {'sea animal': {'aquatic mammals': ['beaver', 'dolphin'],
    #                                   'fish': ['aquarium_fish', 'flatfish']},
    #                    'people': ['baby', 'boy', 'girl', 'man'],
    #                    'flora': {'flowers': ['orchid', 'poppy', 'tulip'],
    #                              'fruit and vegetables': ['apple', 'mushroom']
    #                              }}

    # # assegno la somma delle leaves ai nodi maggiori e i valori singoli ai leaf, mantendendo la batch size
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


def return_matrixes(tree, plot=False):
    matrixes = []
    # read each level and put all the classes name in separate lists
    all_leaves = [leaf.name for leaf in tree.leaves]
    all_nodes = [[node.name for node in children] for children in LevelOrderGroupIter(tree)][1:]

    # build one matrix for each layer that is not the last (we dont a matrix for the last)
    for node_layer in all_nodes[:-1]:

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
            fig = px.imshow(matrix, text_auto=True, aspect="auto", x=node_layer, y=all_leaves, width=2500 // 4,
                            height=2500//4)
            fig.update_xaxes(side="top")
            fig.show()

    return matrixes


if __name__ == "__main__":
    pass
    # tree = get_tree_CIFAR()
    # all_nodes = [node.name for node in LevelOrderIter(tree)][1:]
    # matrix = np.zeros((len(all_nodes), len(all_nodes)), dtype=int)
    #
    # for i, node_name in enumerate(all_nodes):
    #     actual_node = find(tree, lambda node: node.name == node_name)
    #     childrens = actual_node.children
    #     childrens_name = []
    #     for children in childrens:
    #         childrens_name.append(children.name)
    #     for children_name in childrens_name:
    #         index = all_nodes.index(children_name)
    #         matrix[index][i] = 1
    #
    # np.set_printoptions(threshold=sys.maxsize)
    # fig = px.imshow(matrix, text_auto=True, aspect="auto", x=all_nodes, y=all_nodes, width=2500, height=2500)
    # fig.show()

