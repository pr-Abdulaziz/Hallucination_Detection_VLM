from __future__ import annotations

import torch
from typing import Literal

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
        adaptive_weights: torch.Tensor | None = None,
        pair_confidences: torch.Tensor | None = None,
        severity_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if adaptive_weights is not None:
            weight_tensor = adaptive_weights.to(device=losses.device, dtype=losses.dtype)
            return (losses * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)

        if pair_confidences is None or severity_weights is None:
            return losses.mean()

        weights = []
        for pair_confidence, severity_weight in zip(pair_confidences, severity_weights):
            weights.append(adaptive_example_weight(float(pair_confidence), float(severity_weight)))

        weight_tensor = torch.tensor(weights, device=losses.device, dtype=losses.dtype)
        return (losses * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)

    def get_batch_metrics(
        self,
        inputs,
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_scores,
            rejected_scores,
        ) = self.concatenated_forward(self.model, inputs)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, inputs)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_scores,
            rejected_scores,
        )
        pair_confidences = inputs.get("pair_confidences")
        severity_weights = inputs.get("severity_weights")
        adaptive_weights = inputs.get("adaptive_weights")
        loss = self.reduce_weighted_losses(
            losses,
            adaptive_weights=adaptive_weights,
            pair_confidences=pair_confidences,
            severity_weights=severity_weights,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"policy_{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits
        if adaptive_weights is not None:
            metrics[f"{prefix}adaptive/weight_mean"] = adaptive_weights.detach().cpu().mean()
        elif pair_confidences is not None and severity_weights is not None:
            metrics[f"{prefix}adaptive/pair_confidence_mean"] = pair_confidences.detach().cpu().mean()
            metrics[f"{prefix}adaptive/severity_weight_mean"] = severity_weights.detach().cpu().mean()

        return loss, metrics
