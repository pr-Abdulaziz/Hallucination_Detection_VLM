from __future__ import annotations

import torch

from fg_pipeline.adaptive_dpo.adaptive_loss import adaptive_example_weight
from hsa_dpo.trainer.llava_dpo_trainer import LlavaDPOTrainer


class AdaptiveLlavaDPOTrainer(LlavaDPOTrainer):
    """Small extension hook for later adaptive DPO integration.

    This class deliberately avoids changing the original trainer behavior.
    It only adds a helper for weighted example reduction so future work can
    wire pair-level confidence and severity into the final loss.
    """

    def reduce_weighted_losses(
        self,
        losses: torch.Tensor,
        pair_confidences: torch.Tensor | None = None,
        severity_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pair_confidences is None or severity_weights is None:
            return losses.mean()

        weights = []
        for pair_confidence, severity_weight in zip(pair_confidences, severity_weights):
            weights.append(adaptive_example_weight(float(pair_confidence), float(severity_weight)))

        weight_tensor = torch.tensor(weights, device=losses.device, dtype=losses.dtype)
        return (losses * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)
