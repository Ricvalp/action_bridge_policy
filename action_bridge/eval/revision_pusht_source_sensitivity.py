"""Counterfactual source changes at one fixed Push-T policy context.

Every intervention uses the same sampler random numbers: the differences are
responses to the input source, not independently redrawn Brownian noise. These
are coupled command trajectories, not physical rollouts or a distribution test.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from tqdm.auto import tqdm

from action_bridge.data.pusht_adapter import normalize_observations_np
from action_bridge.eval.revision_pusht import validate_evaluation_completion
from action_bridge.plan_revision.cache import reference_for
from action_bridge.plan_revision.checkpoints import frozen_copy
from action_bridge.plan_revision.contracts import ActionCodec
from action_bridge.plan_revision.tracking import preserve_rng
from action_bridge.plan_revision.training import restore_completion


SOURCE_INTERVENTION_FAMILIES = ("translate_x", "translate_y", "bow_x", "bow_y")
DEFAULT_AMPLITUDES_PX = (-40., -15., -5., 5., 15., 40.)


def build_source_interventions(context, config, *, amplitudes_px=DEFAULT_AMPLITUDES_PX,
                               families=SOURCE_INTERVENTION_FAMILIES, region="all"):
    """Edit the actual completed-and-perturbed sampler input, in pixel units.

    The original source noise stays fixed. Observations, action history, previous
    plan and completion are not recomputed. A bow preserves the selected region's
    endpoints and has maximum displacement equal to the requested amplitude.
    """
    source = np.asarray(context["fixed_source_raw"], dtype=np.float32)
    horizon = int(config["horizon"])
    if source.shape != (horizon, 2) or not np.isfinite(source).all():
        raise ValueError("fixed_source_raw must be a finite [H, 2] chunk")
    if not 0 < config["execute"] <= horizon:
        raise ValueError("Require 0 < execute <= horizon")
    executed = int(context["executed"])
    if not 0 <= executed <= horizon:
        raise ValueError("The old plan's executed count must lie in [0, H]")
    retained = horizon - executed if context["has_previous_plan"] else 0
    regions = {"all": np.arange(horizon), "retained": np.arange(retained),
               "tail": np.arange(retained, horizon)}
    if region not in regions:
        raise ValueError("region must be all, retained or tail")
    indices = regions[region]
    if not len(indices):
        raise ValueError(f"The selected {region} region is empty at this decision")
    families = tuple(families)
    if not families or len(set(families)) != len(families) or any(
            family not in SOURCE_INTERVENTION_FAMILIES for family in families):
        raise ValueError(f"Choose unique intervention families from {SOURCE_INTERVENTION_FAMILIES}")
    if any(family.startswith("bow_") for family in families) and len(indices) < 3:
        raise ValueError("A bow requires at least three targets in the selected region; use translations")
    amplitudes = np.asarray(tuple(amplitudes_px), dtype=np.float64)
    if amplitudes.ndim != 1 or not np.isfinite(amplitudes).all():
        raise ValueError("amplitudes_px must be a finite list of signed displacements")
    amplitudes = list(dict.fromkeys(float(value) for value in amplitudes if value != 0))
    if not amplitudes:
        raise ValueError("Provide at least one nonzero displacement")

    sources, labels, names, sizes = [source.copy()], ["baseline"], ["baseline"], [0.]
    for family in families:
        profile = np.ones(len(indices), dtype=np.float32)
        if family.startswith("bow_"):
            profile = np.sin(np.linspace(0., np.pi, len(indices))).astype(np.float32)
            profile[[0, -1]] = 0.
            profile /= profile.max()
        dimension = 0 if family.endswith("_x") else 1
        for amplitude in amplitudes:
            shifted = source.copy()
            shifted[indices, dimension] += amplitude * profile
            if not np.isfinite(shifted).all():
                raise ValueError("Source displacement produced nonfinite targets")
            sources.append(shifted)
            labels.append(f"{family} {amplitude:+g}px")
            names.append(family)
            sizes.append(amplitude)
    return {**context, "sources_raw": np.stack(sources), "variant_labels": labels,
            "families": names, "amplitudes_px": np.asarray(sizes), "baseline_index": 0,
            "intervention_region": region, "retained_count": retained,
            "execute": int(config["execute"])}


@torch.no_grad()
def sample_source_sensitivity(policy, config, metadata, dependencies, context, device="cpu", *,
                              samples=64, batch_size=32, seed=42, completion_id=2,
                              amplitudes_px=DEFAULT_AMPLITUDES_PX,
                              families=SOURCE_INTERVENTION_FAMILIES, region="all", progress=True):
    """Sample all source variants using paired noise and an identical batch layout.

    The frozen reference is computed only from the unchanged histories and reused
    for every source. Resetting the private generator for each variant pairs both
    Brownian increments and kinetic initial auxiliary velocities. Batch size is
    part of reproducibility. No source perturbations are redrawn here.
    """
    if samples < 1 or batch_size < 1 or not 0 <= seed < 2**63 - 1:
        raise ValueError("Require positive samples/batch_size and a nonnegative 63-bit seed")
    if config["method"] not in ("fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"):
        raise ValueError("Source sensitivity requires an FM or SB checkpoint")
    validate_evaluation_completion(config, completion_id)
    data = build_source_interventions(context, config, amplitudes_px=amplitudes_px,
                                      families=families, region=region)
    codec = ActionCodec(**metadata["codec"])
    horizon = int(config["horizon"])
    counts = [min(batch_size, samples - offset) for offset in range(0, samples, batch_size)]
    started = time.perf_counter()
    with preserve_rng():
        sampler = frozen_copy(policy).to(device)
        obs = torch.as_tensor(normalize_observations_np(context["obs_history_raw"], metadata["normalization"]),
                              dtype=torch.float32, device=device)[None]
        actions = codec.encode(torch.as_tensor(context["action_history_raw"], dtype=torch.float32,
                                               device=device))[None]
        references = dict.fromkeys(counts)
        if config["method"].startswith("sb_"):
            completion = restore_completion(dependencies, device)
            prior = completion.prior(
                obs, actions, torch.as_tensor(dependencies["innovation_variance"], device=device),
                ridge=config["prior_ridge"], max_rate=config["max_rate"],
                mobility_smoothing=config["mobility_smoothing"],
            )
            for count in references:
                reference_batch = {"precision": prior["precision"].expand(count, -1, -1),
                                   "prior_mean": prior["mean"].expand(count, -1)}
                if config["mobility_smoothing"]:
                    reference_batch["mobility"] = prior["mobility"]
                references[count] = reference_for(reference_batch, config)

        revision_times = None
        with tqdm(total=(len(data["sources_raw"]) + 2) * samples, desc="Source sensitivity",
                  unit="chunk", disable=not progress) as bar:
            def sample_one(raw_source, noise_seed, *, keep_trace):
                nonlocal revision_times
                generator = torch.Generator(device=device).manual_seed(noise_seed)
                source = codec.encode(torch.as_tensor(raw_source, dtype=torch.float32, device=device))[None]
                plans, traces = [], []
                for count in counts:
                    generated, diagnostics = sampler.sample(
                        obs.expand(count, -1, -1), actions.expand(count, -1, -1), completion_id,
                        source_actions=source.expand(count, -1, -1).clone(), reference=references[count],
                        generator=generator, trace=True,
                        has_previous_plan=torch.full((count,), context["has_previous_plan"],
                                                     device=device, dtype=torch.bool),
                        **({"execution_k": torch.full((count,), config["execute"], device=device,
                                                      dtype=torch.long)}
                           if config.get("condition_on_k", False) else {}),
                    )
                    if generated.shape != (count, horizon, 2) or not torch.isfinite(generated).all():
                        raise ValueError("Sampler returned nonfinite or incorrectly shaped targets")
                    if "revision_states" not in diagnostics or "revision_times" not in diagnostics:
                        raise ValueError("Sampler must return actual revision_states and revision_times")
                    states = torch.as_tensor(diagnostics["revision_states"], device=device)
                    times = torch.as_tensor(diagnostics["revision_times"], device=device)
                    if (times.ndim != 1 or len(times) < 2 or not torch.isfinite(times).all()
                            or times[0] != 0 or times[-1] != 1 or not (times[1:] > times[:-1]).all()
                            or states.shape not in ((count, len(times), horizon * 2),
                                                    (count, len(times), horizon, 2))
                            or not torch.isfinite(states).all()):
                        raise ValueError("Invalid revision trace positions or time grid")
                    states = states.reshape(count, len(times), horizon, 2)
                    if (not torch.allclose(states[:, 0].to(source), source.expand(count, -1, -1), atol=1e-6)
                            or not torch.allclose(states[:, -1].to(generated), generated, atol=1e-6)):
                        raise ValueError("Revision trace must start at the source and end at the generated chunk")
                    times_raw = times.cpu().numpy()
                    if revision_times is None:
                        revision_times = times_raw
                    elif not np.array_equal(revision_times, times_raw):
                        raise ValueError("All variants must use the same revision time grid")
                    plans.append(codec.decode(generated).cpu().numpy())
                    if keep_trace:
                        traces.append(codec.decode(states).cpu().numpy())
                    bar.update(count)
                return np.concatenate(plans), np.concatenate(traces) if keep_trace else None

            plans, traces = [], []
            for source in data["sources_raw"]:
                generated, states = sample_one(source, seed, keep_trace=True)
                plans.append(generated)
                traces.append(states)
            repeated, _ = sample_one(data["sources_raw"][0], seed, keep_trace=False)
            independent, _ = sample_one(data["sources_raw"][0], seed + 1, keep_trace=False)

    return {**data, "candidates_raw": np.stack(plans), "revision_states_raw": np.stack(traces),
            "revision_times": revision_times, "repeat_baseline_raw": repeated,
            "independent_baseline_raw": independent, "method": config["method"],
            "samples": samples, "batch_size": batch_size, "sampling_seed": seed,
            "independent_sampling_seed": seed + 1, "completion_id": completion_id,
            "source_noise_mode": "fixed", "sample_seconds": time.perf_counter() - started}


def sensitivity_statistics(data):
    """RMS Euclidean command differences in unclipped pixels, averaged over N,H.

    A small paired response indicates local insensitivity for this context and
    tested perturbations. It does not prove the conditional output distributions
    are identical, nor that any differing commands would improve task success.
    """
    def rms(displacement):
        return float(np.sqrt(np.mean(np.sum(displacement**2, axis=-1))))

    plans = np.asarray(data["candidates_raw"], dtype=np.float64)
    sources = np.asarray(data["sources_raw"], dtype=np.float64)
    baseline_index = int(data["baseline_index"])
    baseline, source = plans[baseline_index], sources[baseline_index]
    variants = []
    for index, label in enumerate(data["variant_labels"]):
        delta = plans[index] - baseline
        source_shift = rms(sources[index] - source)
        output_shift = rms(delta)
        variants.append({
            "label": label, "family": data["families"][index],
            "amplitude_px": float(data["amplitudes_px"][index]),
            "source_rms_shift_px": source_shift,
            "paired_output_rms_shift_px": output_shift,
            "gain": output_shift / source_shift if source_shift > 0 else None,
            "mean_output_shift_px": rms(delta.mean(axis=0)),
            "executed_prefix_output_rms_shift_px": rms(delta[:, :data["execute"]]),
            "kth_output_rms_shift_px": rms(delta[:, data["execute"] - 1]),
            "last_output_rms_shift_px": rms(delta[:, -1]),
            "output_rms_spread_px": rms(plans[index] - plans[index].mean(axis=0)),
        })
    return {"samples": int(data["samples"]), "variants": variants,
            "baseline_output_rms_spread_px": rms(baseline - baseline.mean(axis=0)),
            "repeat_same_noise_rms_px": rms(np.asarray(data["repeat_baseline_raw"], dtype=np.float64) - baseline),
            "same_source_independent_noise_rms_px": rms(
                np.asarray(data["independent_baseline_raw"], dtype=np.float64) - baseline),
            "units": "unclipped command-target pixels; Euclidean RMS across samples and chunk targets"}
