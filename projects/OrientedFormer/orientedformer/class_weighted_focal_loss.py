import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.registry import MODELS
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
from mmdet.models.losses.utils import weight_reduce_loss


@MODELS.register_module()
class ClassWeightedFocalLoss(nn.Module):
    """Focal Loss with per-class weights for imbalanced datasets.

    Wraps the standard sigmoid focal loss and scales the per-class loss
    by ``class_weight[c]`` before reduction.

    Args:
        use_sigmoid (bool): Must be True (sigmoid focal loss).
        gamma (float): Focusing parameter. Default: 2.0.
        alpha (float): Balance factor for pos/neg. Default: 0.25.
        class_weight (list[float] | None): Per-class multiplicative weights,
            length must equal num_classes. Default: None (uniform).
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'.
        loss_weight (float): Overall scalar on the loss. Default: 1.0.
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 class_weight=None,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        assert use_sigmoid, 'Only sigmoid focal loss supported.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        if class_weight is not None:
            self.register_buffer(
                'class_weight',
                torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override or self.reduction

        # target may be (N,) indices or (N, C) one-hot / soft
        if pred.dim() != target.dim():
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes].float()

        # clamp logits to prevent BCE exploding when avg_factor is tiny
        pred = pred.clamp(-20.0, 20.0)

        # floor avg_factor so a batch with 1 positive doesn't divide by ~1
        # while summing 300*C loss terms
        if avg_factor is not None:
            avg_factor = max(avg_factor, 10.0)

        # compute per-element focal loss  (N, C)
        loss = py_sigmoid_focal_loss(
            pred, target, weight=None,
            gamma=self.gamma, alpha=self.alpha,
            reduction='none', avg_factor=None)

        # apply per-class weight  (1, C) * (N, C) → (N, C)
        if self.class_weight is not None:
            cw = self.class_weight.to(loss.device)
            loss = loss * cw.unsqueeze(0)

        # apply per-sample weight  (N,) → (N, 1) for broadcast
        if weight is not None:
            if weight.dim() == 1 and loss.dim() == 2:
                weight = weight.unsqueeze(1)

        # reduce
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction,
            avg_factor=avg_factor)
        return self.loss_weight * loss
