from __future__ import division, print_function, absolute_import

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .diversity_loss import DiversityLoss

def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    
    loss = 0.

    if isinstance(criterion, DiversityLoss):
        return criterion(xs, y)
    
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
