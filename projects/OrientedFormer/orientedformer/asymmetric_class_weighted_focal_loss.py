import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss


@MODELS.register_module()
class AsymmetricClassWeightedFocalLoss(nn.Module):
    """Positive-only class-weighted focal loss (Strategy B).

    Differs from ``ClassWeightedFocalLoss`` in one principled way:
    per-class weights are applied **only to the positive term**
    (``target == 1``). Negative terms (``target == 0``) keep neutral
    weight 1.0. This amplifies "push up" gradients for rare/priority
    classes without amplifying "this is not class X" gradients across
    every other query in the batch.

    Optionally supports two well-known refinements:

    * Asymmetric Loss (Ben-Baruch et al., 2020): distinct ``gamma_pos``
      and ``gamma_neg`` decouple positive emphasis from negative
      suppression; ``clip`` hard-zeros the loss on very-easy negatives
      via a probability margin shift.
    * Class-Balanced Loss (Cui et al., CVPR 2019): per-class weights are
      derived from the effective number of samples
      ``(1 - beta) / (1 - beta**n_c)`` when ``class_counts`` is given.

    Defaults reproduce standard sigmoid focal loss with positive-only
    class weighting, i.e. the minimal Strategy-B change.

    Args:
        use_sigmoid (bool): Must be True. Only sigmoid focal supported.
        gamma (float): Default focusing parameter, used when
            ``gamma_pos`` / ``gamma_neg`` are None. Default: 2.0.
        gamma_pos (float | None): Focusing on hard positives. If None,
            falls back to ``gamma``. ASL preset: 0.0.
        gamma_neg (float | None): Focusing on hard negatives. If None,
            falls back to ``gamma``. ASL preset: 4.0.
        alpha (float): Pos/neg balance factor. Default: 0.25.
        clip (float): Probability margin for negatives (ASL). The
            negative term uses ``max(prob - clip, 0)`` so very-easy
            negatives contribute zero loss. 0 disables. Default: 0.0.
        class_weight (list[float] | None): Per-class positive
            multiplicative weights, length must equal ``num_classes``.
            Mutually exclusive with ``class_counts``.
        class_counts (list[int] | None): Per-class sample counts. If
            given, class-balanced weights are computed automatically and
            ``class_weight`` must be None.
        cb_beta (float): Class-balance hyperparameter; typical 0.999 or
            0.9999. Larger beta → stronger reweighting of rare classes.
            Default: 0.999.
        cb_normalize (str): How to normalise class-balanced weights.
            'mean' rescales so the weights' mean is 1 (preserves overall
            gradient magnitude). 'none' leaves them as-is. Default: 'mean'.
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'.
        loss_weight (float): Overall scalar on the loss. Default: 1.0.
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 gamma_pos=None,
                 gamma_neg=None,
                 alpha=0.25,
                 clip=0.0,
                 class_weight=None,
                 class_counts=None,
                 cb_beta=0.999,
                 cb_normalize='mean',
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        assert use_sigmoid, 'Only sigmoid focal loss supported.'
        assert not (class_weight is not None and class_counts is not None), \
            'Pass class_weight OR class_counts, not both.'
        assert cb_normalize in ('mean', 'none')

        self.use_sigmoid = use_sigmoid
        self.gamma_pos = gamma if gamma_pos is None else gamma_pos
        self.gamma_neg = gamma if gamma_neg is None else gamma_neg
        self.alpha = alpha
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight

        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float64)
            eff_num = 1.0 - torch.pow(torch.tensor(cb_beta, dtype=torch.float64),
                                      counts)
            weights = (1.0 - cb_beta) / eff_num.clamp(min=1e-12)
            if cb_normalize == 'mean':
                weights = weights / weights.mean()
            cw = weights.float()
        elif class_weight is not None:
            cw = torch.tensor(class_weight, dtype=torch.float32)
        else:
            cw = None

        if cw is not None:
            self.register_buffer('class_weight', cw)
        else:
            self.class_weight = None

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override or self.reduction

        # target may be (N,) class indices or (N, C) one-hot / soft
        if pred.dim() != target.dim():
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes].float()
        target = target.type_as(pred)

        # numerically stable log-sigmoid forms
        log_p = F.logsigmoid(pred)        # log(sigmoid(x))
        log_1mp = F.logsigmoid(-pred)     # log(1 - sigmoid(x))
        prob = pred.sigmoid()

        # ASL probability margin on the negative term
        if self.clip > 0:
            shifted = (1.0 - prob + self.clip).clamp(min=1e-8, max=1.0)
            log_1mp_neg = torch.log(shifted)
            prob_neg = (prob - self.clip).clamp(min=0.0)
        else:
            log_1mp_neg = log_1mp
            prob_neg = prob

        pos_focal = (1.0 - prob).pow(self.gamma_pos)
        neg_focal = prob_neg.pow(self.gamma_neg)

        pos_term = self.alpha * pos_focal * (-log_p) * target
        neg_term = (1.0 - self.alpha) * neg_focal * (-log_1mp_neg) * (1.0 - target)

        # Strategy B: per-class weight applied ONLY to the positive term
        if self.class_weight is not None:
            cw = self.class_weight.to(pos_term.device).unsqueeze(0)
            pos_term = pos_term * cw

        loss = pos_term + neg_term

        # per-sample weight broadcast over classes
        if weight is not None and weight.dim() == 1 and loss.dim() == 2:
            weight = weight.unsqueeze(1)

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction,
            avg_factor=avg_factor)
        return self.loss_weight * loss
