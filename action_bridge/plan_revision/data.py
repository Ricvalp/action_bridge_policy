"""Build frozen proposal pairs from a dataset-supplied common window batch."""
from __future__ import annotations

import torch

from action_bridge.plan_revision.contracts import take, tree_map


@torch.no_grad()
def build_source_pairs(windows, proposal, completion, innovation_variance, config, device):
    """Never execute a proposal, replace observations or inspect expert futures.

    A record keeps its original labels, two earlier DDIM predictions and frozen
    observed-history reference coefficients. Tail mode is drawn later so the
    three modes always use exactly the same underlying prediction distribution.
    """
    count = len(windows["future_actions"])
    indices = torch.arange(count).repeat_interleave(config["proposals_per_history"])
    records = take(windows, indices)
    predictions, precisions, means = [], [], []
    generator = torch.Generator(device=device).manual_seed(config["proposal_seed"])
    for start in range(0, len(indices), config["batch_size"]):
        batch = take(records, slice(start, start + config["batch_size"]), device)
        actions, _ = proposal.sample(batch["earlier_obs_hist"], batch["earlier_act_hist"], generator=generator)
        prior = completion.prior(batch["obs_hist"], batch["act_hist"],
                                 innovation_variance.to(device),
                                 ridge=config["prior_ridge"], max_rate=config["max_rate"],
                                 mobility_smoothing=config["mobility_smoothing"])
        predictions.append(actions.cpu())
        precisions.append(prior["precision"].cpu())
        means.append(prior["mean"].cpu())
    records.update(old_actions=torch.cat(predictions), precision=torch.cat(precisions),
                   prior_mean=torch.cat(means), proposal_id=torch.arange(len(indices)) %
                   config["proposals_per_history"])
    # Identity mobility is implicit, enabling the fast modal Gaussian backend.
    if config["mobility_smoothing"]:
        records["mobility"] = prior["mobility"].cpu().expand(len(indices), -1, -1).clone()
    return tree_map(lambda x: x.detach().cpu(), records)
