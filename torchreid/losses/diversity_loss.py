from __future__ import division, absolute_import
import torch
import torch.nn as nn


class DiversityLoss(nn.Module):
    """Diversity Loss to enforce diversity among part based embeddings

       Reference: Li, Yulin, et al. Diverse Part Discovery: Occluded Person Re-Identification With Part-Aware Transformer.

       Args:
        n_templates: Number of part prototypes to be predicted
    """

    def __init__(self, n_templates=2):
        super().__init__()
        self.K = n_templates
        self.loss_scale = 1./((self.K-1)*self.K)

    def forward(self, inputs, targets):

        if isinstance(inputs, (tuple, list)):
            assert len(inputs) == self.K, f'inputs {len(inputs)} != templates {self.K}'
            inputs = torch.stack(inputs).transpose(0, 1)                                     # stacks the part-based embeddings [bs, n_temp, d]
            dots = inputs @ inputs.transpose(2, 1)                                           # taking dot product between each of the vectors [bs, n_temp, n_temp]
            norms = torch.sqrt(torch.diagonal(dots, 0, dim1=-2, dim2=-1)).unsqueeze(2)      # taking the L2 norm values (the main diagonal and square root of it) [bs, 1, n_temp]
            norms = torch.matmul(norms, norms.transpose(2, 1))                               # matrix multiplication to obatin the L2 norm multipliers to divide by [bs, n_temp, n_temp]
            dots = dots / norms                                                             # divide each dot product element by the corresponding norm multipliers
            dots = torch.triu(dots, 1) + torch.tril(dots, -1)                               # removing the main diagonal (i!=j)
            dots = torch.sum(dots, dim=(1, 2))*self.loss_scale                               # diverse loss for a single sample

            return torch.mean(dots)         # returning the loss value as the mean value over the batch size
        else:
            raise TypeError('Invalid type of input. Should be a tuple or list.')


def unit_test(input_vector):
    """
        This is to verify the diverse loss class using the normal for-loop
        method.
    """
    input_vector = [tensor.numpy() for tensor in input_vector]
    input_vector = torch.as_tensor(input_vector).transpose(0, 1)

    b, k, d = input_vector.shape
    loss_values = 0
    for i in range(b):
        sample = input_vector[i]
        loss = 0
        for j in range(k):
            vector1 = sample[j, :]
            norm1 = torch.norm(vector1)
            for jj in range(k):
                if j == jj:
                    continue
                vector2 = sample[jj, :]
                norm2 = torch.norm(vector2)
                dot = vector1 @ vector2
                loss += dot/(norm1*norm2)
        loss_values += loss/(k*(k-1))
    return loss_values/b


if __name__ == '__main__':
    input_list = []
    n_templates = 8
    bs = 32
    dim = 768
    for _ in range(n_templates):
        input_list.append(torch.rand(bs, dim))

    losses = unit_test(input_list)
    print(losses)

    criteria = DiversityLoss(n_templates=n_templates)
    loss_func = criteria(input_list, None)
    print(loss_func)
