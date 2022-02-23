import numpy as np
import torch.nn.functional as F
import torch


predicted = torch.tensor([[0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1], [0.3, 0.4, 0.2, 0.1], [0.3, 0.2, 0.1, 0.4]], dtype=torch.float32)
actual = torch.tensor([1, 1, 2, 3], dtype=torch.int64)

# classes = [0, 1, 2, 3]
coarse_labels = np.array([0, 0, 1, 1])

# compute the loss for fine classes
loss_fine = F.cross_entropy(predicted, actual)

superclasses = 2
# define an empty vector which contains 20 superclasses prediction for each samples
predicted_coarse = torch.zeros(4, superclasses, dtype=torch.float32, device="cuda:0")

# for each samples
# for i, sample in enumerate(predicted):
    # for each superclass
for k in range(superclasses):
    # obtain the indexes of the superclass number k
    indexes = list(np.where(coarse_labels == k))[0]
    # for each index, sum all the probability related to that superclass
    #predicted_coarse[i][k] = torch.sum(predicted[i][indexes])
    # this is the same as doing:
    for j in indexes:
        a = predicted[:, j]
        b = predicted_coarse[:, k].cpu()
        predicted_coarse[:, k] = a + b


actual_coarse = sparse2coarse(actual.cpu().detach().numpy())

loss_coarse = F.cross_entropy(predicted_coarse, torch.from_numpy(coarse_labels).type(torch.int64).to("cuda:0"))

print(loss_coarse)