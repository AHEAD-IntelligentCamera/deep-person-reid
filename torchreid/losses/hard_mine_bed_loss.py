from __future__ import division, absolute_import
import torch
import torch.nn as nn


class BEDLoss(nn.Module):
    """
    Bounded Exponential Distance Loss
    """

    def __init__(self, alpha=0.3):
        super(BEDLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute bed loss
        pos_loss = torch.mean(torch.exp(- self.alpha * dist_ap))
        neg_loss = torch.mean(torch.exp(- self.alpha * dist_an))
        return (1 - pos_loss) + neg_loss
