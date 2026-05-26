import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.registry import MODELS
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
from mmdet.models.losses.utils import weight_reduce_loss


@MODELS.register_module()
class FocalLossDDQAdapter(nn.Module):
    """Standard sigmoid focal loss wrapped in the QualityFocalLoss call signature.

    ``OrientedAdaMixerDDQ`` inherits from ``OrientedDDQFCN``, whose
    ``loss_single`` is hard-wired to call the classification loss as::

        loss_cls(score, (labels, alignment_metrics), label_weights, avg_factor=1.0)

    ``QualityFocalLoss`` consumes the alignment metric as a soft
    IoU-quality target. This adapter accepts the same tuple interface,
    **discards the alignment metric**, builds a hard one-hot target
    from labels (background index = ``num_classes``), and computes
    standard sigmoid focal loss — without editing the parent class.

    IMPORTANT — ``activated``:
      The OrientedDDQ RPN outputs ``cls_score = sigmoid_geometric_mean(...)``,
      i.e. **already-activated probabilities in [0, 1]**, NOT raw logits.
      The baseline QFL uses ``activated=True`` to skip its internal sigmoid.
      This adapter mirrors that convention and defaults to ``activated=True``.
      If you flip it to ``False``, you must also flip the model's output
      to be raw logits — otherwise easy negatives get treated as
      "uncertain" predictions and the loss explodes by 100-1000×.

    Args:
        use_sigmoid (bool): Must be True. Only sigmoid focal supported.
        activated (bool): True if ``pred`` is already a probability.
            False if ``pred`` is a raw logit. Default: True (matches
            OrientedDDQ RPN output).
        gamma (float): Focusing parameter. Default: 2.0.
        alpha (float): Pos/neg balance factor. Default: 0.25.
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'.
        loss_weight (float): Overall scalar on the loss. Default: 1.0.
    """

    def __init__(self,
                 use_sigmoid=True,
                 activated=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        assert use_sigmoid, 'Only sigmoid focal loss supported.'
        self.use_sigmoid = use_sigmoid
        self.activated = activated
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        # DDQ-FCN passes target as (labels, alignment_metrics). Drop the
        # alignment metric to recover plain (hard 0/1) focal semantics.
        if isinstance(target, (tuple, list)):
            labels = target[0]
        else:
            labels = target

        num_classes = pred.size(-1)
        if labels.dim() != pred.dim():
            # labels: (N,) class indices; BG index is num_classes (no column)
            one_hot = F.one_hot(labels.long(), num_classes=num_classes + 1)
            target_hot = one_hot[..., :num_classes].type_as(pred)
        else:
            target_hot = labels.type_as(pred)

        reduction = reduction_override or self.reduction

        if self.activated:
            # pred is already a probability — compute focal loss directly
            # without re-applying sigmoid. Clamp guards log(0) on saturation.
            eps = 1e-8
            prob = pred.clamp(min=eps, max=1.0 - eps)
            pt = (1.0 - prob) * target_hot + prob * (1.0 - target_hot)
            alpha_t = self.alpha * target_hot + (1.0 - self.alpha) * (1.0 - target_hot)
            focal_weight = alpha_t * pt.pow(self.gamma)
            bce = -(target_hot * torch.log(prob)
                    + (1.0 - target_hot) * torch.log(1.0 - prob))
            loss = focal_weight * bce

            # mirror py_sigmoid_focal_loss's weight broadcasting
            if weight is not None and weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)

            loss = weight_reduce_loss(
                loss, weight=weight, reduction=reduction,
                avg_factor=avg_factor)
        else:
            # pred is raw logits — delegate to mmdet's stable kernel
            loss = py_sigmoid_focal_loss(
                pred, target_hot,
                weight=weight,
                gamma=self.gamma, alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        return self.loss_weight * loss
