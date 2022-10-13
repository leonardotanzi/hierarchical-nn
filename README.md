# hierarchical-nn

## Implementation of a hierarchical neural network for bone fracture classification

CNN_bones.py: implementation of the hierarchical loss (hl) applied to fracture
CNN_test_bones.py: testing for bones 

CNN_CIFAR100.py: first implementation of hl applied to CIFAR100
CNN_CIFAR100-ExcludeClass.py: implementation of hl applied to a subset of CIFAR100
CNN_CIFAR100-Regularization.py: implementation of regularization proposed in the paper from Neal and Shahbaba

CNN_CIFAR100-IntermediatePrediction.py: added a second branch after the second conv layer for an intermediate superclass prediction

CNN_test_CIFAR100.py: testing and confusion matrixes for CIFAR

LossUnderstanding.py: small script to understand the hl

utils.py: some function and classes
