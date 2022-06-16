from anytree import Node, RenderTree, LevelOrderGroupIter, NodeMixin
import torch


class MyNode(Node):  # Add Node feature
    def __init__(self, name, prob_value, parent=None, children=None):
        super(MyNode, self).__init__(name, parent=None, children=None)
        self.name = name
        self.parent = parent
        self.loss_value = prob_value
        if children:  # set children only if given
            self.children = children


def get_tree_CIFAR(outputs):

    superclass_dict = {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                       'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                       'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                       'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
                       'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                       'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                       'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                       'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                       'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                       'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                       'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                       'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                       'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                       'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
                       'people': ['baby', 'boy', 'girl', 'man', 'woman'],
                       'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                       'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                       'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                       'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                       'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}

    predicted = torch.softmax(outputs, dim=1) + 1e-6

    # assegno la somma delle leaves ai nodi maggiori e i valori singoli ai leaf, mantendendo la batch size
    root = MyNode("root", 0.0)
    for i, (k, v) in enumerate(superclass_dict.items()):
        prob = predicted[:, i*5:i*5+5]
        p = MyNode(f"{k}", torch.sum(prob, dim=1), parent=root)
        for j, c in enumerate(v):
            n = MyNode(f"{c}", prob[:, j], parent=p)

    print(RenderTree(root))


if __name__ == "__main__":
    pass