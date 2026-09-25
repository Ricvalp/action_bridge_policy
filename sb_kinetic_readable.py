"""Reading version of SB-kinetic training. Start at train_sb_kinetic().

Same learning steps as the real trainer; no logging, evaluation, checkpoint
saving, resume logic, or exact-RNG reproduction. All actions are normalized.

Optional execution with your own *trusted*, prepared H=16 / K=8 artifacts:
    python sb_kinetic_readable.py /path/to/prepared_run --device cuda
That directory must contain windows.pt and reference/latest.pt. This actually
does 300,000 updates and saves NOTHING: use the normal trainer for experiments.
"""

import argparse
from pathlib import Path

import torch

from action_bridge.plan_revision.checkpoints import frozen_copy, load, update_ema
from action_bridge.plan_revision.contracts import take
from action_bridge.plan_revision.data import build_self_sources
from action_bridge.plan_revision.gaussian import GaussianReference
from action_bridge.plan_revision.models import BridgePolicy
from action_bridge.plan_revision.training import restore_completion


CONFIG = dict(
    method="sb_kinetic", obs_dim=5, action_dim=2,
    obs_history=2, action_history=2, horizon=16, execute=8,
    channels=[80, 160, 320], history_dim=256, hidden_dim=256, time_dim=64,
    batch_size=256, num_inference_steps=32, time_cutoff=0.01,
    robot_dt=1.0, source_std=0.01, prior_ridge=0.05, max_rate=4.0,
    mobility_smoothing=0.0, temperature=0.05, revision_gamma=2.0,
)
SELF_PROBABILITIES = (0.10, 0.50, 1.0, 1.0)
PHASE_UPDATES = 37_500
PAIR_COUNT = 4096
PAIR_REFRESH_EVERY = 1000


def train_sb_kinetic(train_windows, completion, innovation_variance, device="cpu"):
    """4 source blocks x (reverse phase + forward phase) x 37,500 updates."""
    torch.manual_seed(0)
    policy = BridgePolicy(CONFIG).to(device).train()
    ema = frozen_copy(policy)  # Exponential moving average, used for generation.
    completion = frozen_copy(completion).to(device)  # Pretrained; NEVER optimized here.

    fields = {"forward": policy.forward_field, "reverse": policy.reverse_field}
    ema_fields = {"forward": ema.forward_field, "reverse": ema.reverse_field}
    optimizers = {
        name: torch.optim.AdamW(field.parameters(), lr=1e-4, weight_decay=1e-6)
        for name, field in fields.items()
    }
    global_step = 0

    # BLOCK: fix the population of starting chunks for BOTH phases below.
    for block, p_self in enumerate(SELF_PROBABILITIES):
        source_producer = frozen_copy(ema)
        sources, _ = build_self_sources(
            train_windows, source_producer, completion, innovation_variance,
            CONFIG, device, block=block, seed=17 + block, p_self=p_self,
        )
        # This helper replays RECORDED observation histories, not a simulator:
        #   old = previous generated plan with probability p_self, else previous expert plan
        #   source = old[K:] + K newly completed tail actions + small Gaussian noise
        # It uses all 3 modes: repeat / fixed damping / learned dissipative completion.
        # At episode start, there is no old plan: repeat the observed action anchor.
        # Frozen forward generation supplies the next old plan during this replay.

        # PHASE: train ONE network against pairs from the frozen OPPOSITE network.
        for direction in ("reverse", "forward"):
            opposite_snapshot = frozen_copy(ema)
            if block == 0 and direction == "reverse":
                opposite_snapshot = None  # Bootstrap from natural source/expert pairs.
            field = fields[direction]
            optimizer = optimizers[direction]
            pairs = None

            for _ in range(PHASE_UPDATES):
                # Refresh endpoint pairs, NOT the block's source population.
                # The opposite network stays frozen throughout this phase.
                if pairs is None or global_step % PAIR_REFRESH_EVERY == 0:
                    pairs = make_training_pairs(sources, opposite_snapshot, direction, device)

                indices = torch.randint(len(pairs["x0"]), (CONFIG["batch_size"],))
                batch = take(pairs, indices, device)
                reference = make_reference(batch)
                x0, x1 = batch["x0"], batch["x1"]

                # tau is GENERATION time in [0,1], not a robot timestep.
                tau = 0.01 + 0.98 * torch.rand(len(x0), device=device)
                with torch.no_grad():
                    # Exact Gaussian reference bridge conditional on these endpoints.
                    state = reference.sample_bridge(x0, x1, tau)
                    forward_target, reverse_target = reference.controls(state, x0, x1, tau)
                    target = reverse_target if direction == "reverse" else forward_target
                    distance = tau if direction == "reverse" else 1 - tau

                # Predict the noise-channel control u, NOT the next robot action.
                # Endpoint labels x0/x1 are NOT passed as conditioning inputs.
                prediction = field(
                    state, tau, batch["obs_hist"], batch["act_hist"],
                    batch["completion_id"], batch["has_previous_plan"],
                )
                error = (prediction - target).square().mean(dim=-1)
                loss = (distance**3 * error).mean()  # Kinetic endpoint weighting.

                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # No backprop through source replay or pair generation.
                torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
                optimizer.step()
                update_ema(ema_fields[direction], field, decay=0.999)
                global_step += 1

    # Networks and optimizers were carried across phases/blocks, never reset.
    # Inference uses the forward EMA: completed old plan -> revised action chunk.
    return ema


def make_reference(batch):
    """Frozen Gaussian dynamics; mean/precision come from the pretrained completer."""
    return GaussianReference(
        batch["precision"], batch["prior_mean"], kind="kinetic",
        temperature=CONFIG["temperature"], gamma=CONFIG["revision_gamma"],
    )


@torch.no_grad()
def make_training_pairs(sources, opposite_snapshot, direction, device):
    """The alternating step: change one endpoint using the opposite sampler.

    First reverse phase: source ------------------------ expert
    Later reverse phases: source -- frozen forward --> generated endpoint
    Every forward phase:  generated endpoint <-- frozen reverse -- expert

    These are pairs of entire chunks in revision time, not two robot timesteps.
    """
    pieces = []
    for offset in range(0, PAIR_COUNT, CONFIG["batch_size"]):
        count = min(CONFIG["batch_size"], PAIR_COUNT - offset)
        indices = torch.randint(len(sources["source_actions"]), (count,))
        batch = take(sources, indices, device)
        reference = make_reference(batch)

        source = batch["source_actions"]  # Already completed/noised once; do NOT re-noise.
        expert = batch["future_actions"]
        expert = expert + 0.001 * torch.randn_like(expert)  # Smooth the target endpoint law.

        # Kinetic state = [all chunk positions, auxiliary revision velocities].
        # augment adds independent N(0, temperature * I) velocities, NOT robot velocities.
        x0 = reference.augment(source.flatten(1))
        x1 = reference.augment(expert.flatten(1))

        if opposite_snapshot is not None:
            # To TRAIN forward, GENERATE backwards from expert (and vice versa).
            run_reverse = direction == "forward"
            generated, _ = opposite_snapshot.rollout(
                x1 if run_reverse else x0,
                batch["obs_hist"], batch["act_hist"], batch["completion_id"], reference,
                reverse=run_reverse, steps=CONFIG["num_inference_steps"],
                has_previous_plan=batch["has_previous_plan"],
            )
            # Keep the full generated kinetic state, including its velocity.
            if run_reverse:
                x0 = generated
            else:
                x1 = generated

        batch["x0"], batch["x1"] = x0, x1
        pieces.append({key: value.cpu() for key, value in batch.items()})

    return {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_run", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    data = load(args.prepared_run / "windows.pt")
    reference_state = load(args.prepared_run / "reference" / "latest.pt")
    completer = restore_completion(reference_state, args.device)
    trained_policy = train_sb_kinetic(
        data["records"]["train"], completer, reference_state["innovation_variance"], args.device,
    )
