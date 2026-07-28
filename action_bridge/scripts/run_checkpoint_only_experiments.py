"""Run checkpoint-only Push-T experiments for a trained policy.

These experiments do not retrain the model. They reuse one checkpoint and run
closed-loop simulator variants that intervene on the reference, receding horizon,
and latent sampling behavior.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from ml_collections import ConfigDict

from action_bridge.config import apply_overrides, save_config
from action_bridge.data.pusht_adapter import denormalize_actions_tensor, normalize_actions_tensor, normalize_observations_tensor
from action_bridge.eval.rollout import generate_chunk
from action_bridge.eval.pusht_sim import evaluate_pusht_sim_model
from action_bridge.eval.pusht_sim_parallel import evaluate_pusht_sim_checkpoint_parallel
from action_bridge.eval.pusht_wrong_side import _draw_tee, _synthetic_wrong_side_states, plot_wrong_side_go_around_diagnostic
from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.training.common import build_model, resolve_device, save_json, seed_everything


def timestamp_prefix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def checkpoint_run_dir(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parents[1]
    return Path("outputs")


def default_output_dir(checkpoint: Path, name: str | None = None) -> Path:
    run_dir = checkpoint_run_dir(checkpoint)
    label = name or f"checkpoint_only_{checkpoint.stem}"
    return run_dir / "eval" / f"{timestamp_prefix()}_{label}"


def set_nested(config: ConfigDict, dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if key not in target:
            target[key] = ConfigDict()
        target = target[key]
    target[parts[-1]] = value


def make_eval_config(
    base_config: ConfigDict,
    args: argparse.Namespace,
    variant: Dict[str, Any],
) -> ConfigDict:
    config = copy.deepcopy(base_config)
    if args.device is not None:
        config.device = args.device
    if "eval" not in config:
        config.eval = ConfigDict()

    config.eval.sim_closed_loop = True
    config.eval.sim_episodes = int(args.episodes)
    config.eval.sim_seed = int(args.seed)
    config.eval.sim_max_steps = int(args.max_steps)
    config.eval.sim_render_episodes = int(args.render_episodes)
    config.eval.sim_save_gifs = bool(args.save_gifs)
    config.eval.sim_gif_fps = float(args.gif_fps)
    config.eval.sim_save_videos = bool(args.save_videos)
    config.eval.sim_video_fps = float(args.video_fps)
    config.eval.sim_collect_contact_diagnostics = True
    config.eval.sim_latent_samples = int(args.latent_samples)
    config.eval.sim_latent_sample_panels = int(args.latent_sample_panels)
    config.eval.sim_contact_panels = int(args.contact_panels)
    config.eval.sim_contact_grid_size = int(args.contact_grid_size)
    config.eval.wrong_side_go_around = True
    config.eval.wrong_side_num_samples = int(args.wrong_side_samples)
    config.eval.latent_sweep_states = int(args.latent_sweep_states)
    config.eval.latent_sweep_std_scale = float(args.latent_sweep_std_scale)

    for key, value in variant.get("overrides", {}).items():
        set_nested(config, key, value)
    return config


def experiment_variants(base_n_exec: int) -> list[Dict[str, Any]]:
    variants: list[Dict[str, Any]] = []

    reference_interventions = [
        ("full_policy", "full_policy"),
        ("reference_only", "reference_only"),
        ("control_only", "control_only"),
        ("no_damping", "no_damping"),
        ("no_potential", "no_potential"),
    ]
    for name, intervention in reference_interventions:
        variants.append(
            {
                "group": "reference_intervention",
                "name": name,
                "description": f"sim_intervention={intervention}",
                "overrides": {
                    "eval.sim_n_exec": int(base_n_exec),
                    "eval.sim_intervention": intervention,
                    "eval.sim_latent_mode": "default",
                },
            }
        )

    for n_exec in [1, 2, 4, 8, 16]:
        variants.append(
            {
                "group": "receding_horizon",
                "name": f"n_exec_{n_exec}",
                "description": f"full policy with n_exec={n_exec}",
                "overrides": {
                    "eval.sim_n_exec": int(n_exec),
                    "eval.sim_intervention": "full_policy",
                    "eval.sim_latent_mode": "default",
                },
            }
        )

    latent_modes = [
        ("prior_mean", "prior_mean"),
        ("episode_sticky_sample", "episode_sample"),
        ("replan_resampled", "replan_sample"),
    ]
    for name, latent_mode in latent_modes:
        variants.append(
            {
                "group": "latent_causal_use",
                "name": name,
                "description": f"full policy with sim_latent_mode={latent_mode}",
                "overrides": {
                    "eval.sim_n_exec": int(base_n_exec),
                    "eval.sim_intervention": "full_policy",
                    "eval.sim_latent_mode": latent_mode,
                },
            }
        )
    return variants


def parse_variant_filters(raw: str | None) -> set[str]:
    if raw is None or not raw.strip():
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def variant_keys(variant: Dict[str, Any]) -> set[str]:
    group = str(variant["group"])
    name = str(variant["name"])
    return {name, f"{group}/{name}", f"{group}__{name}"}


def filter_variants(variants: list[Dict[str, Any]], raw_filters: str | None) -> list[Dict[str, Any]]:
    filters = parse_variant_filters(raw_filters)
    if not filters:
        return variants
    selected = [variant for variant in variants if variant_keys(variant) & filters]
    matched = set()
    for variant in selected:
        matched.update(variant_keys(variant) & filters)
    missing = sorted(filters - matched)
    if missing:
        available = sorted({key for variant in variants for key in variant_keys(variant)})
        raise ValueError(f"Unknown checkpoint-only variant(s): {missing}. Available variants: {available}")
    return selected


def write_summary_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalization_stats(config: Dict[str, Any]) -> Dict[str, Any] | None:
    data_cfg = config.get("data", {})
    stats = data_cfg.get("normalization_stats")
    if bool(data_cfg.get("normalize", False)) and stats is not None:
        return stats
    return None


@torch.no_grad()
def plot_prior_direction_latent_sweep(model, config: Dict[str, Any], device: torch.device, output_dir: Path) -> Dict[str, float]:
    """Evaluate chunks at z = mu and mu +/- c std e_i for wrong-side histories."""

    if not isinstance(model, ActionBridgePolicy):
        return {}
    if model.latent_type != "continuous" or not hasattr(model.latent, "prior_params"):
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = config.get("eval", {})
    stats = _normalization_stats(config)
    obs_history = int(config.get("obs_history", 2))
    action_history = int(config.get("action_history", 2))
    sweep_scale = float(eval_cfg.get("latent_sweep_std_scale", 2.0))
    max_states = int(eval_cfg.get("latent_sweep_states", 4))
    sources, construction = _synthetic_wrong_side_states(config)
    sources = sources[: max(1, max_states)]
    theta = float(construction["theta"])
    goal_pose = [256.0, 256.0, theta]

    records = []
    fig, axes = plt.subplots(
        len(sources),
        1,
        figsize=(6.2, max(4.2, 4.0 * len(sources))),
        squeeze=False,
        layout="constrained",
    )
    for ax, source in zip(axes.ravel(), sources):
        raw_state = torch.tensor(source["state"], dtype=torch.float32)
        obs_raw = raw_state[None, None, :].expand(1, obs_history, -1).clone()
        act_raw = raw_state[:2][None, None, :].expand(1, action_history, -1).clone()
        obs_hist = normalize_observations_tensor(obs_raw, stats).to(device) if stats is not None else obs_raw.to(device)
        act_hist = normalize_actions_tensor(act_raw, stats).to(device) if stats is not None else act_raw.to(device)

        h_emb = model.encode_history(obs_hist, act_hist)
        mu, logvar = model.latent.prior_params(h_emb)
        std = torch.exp(0.5 * logvar)
        z_values = [("mu", mu)]
        for dim in range(mu.shape[-1]):
            offset = torch.zeros_like(mu)
            offset[:, dim] = sweep_scale * std[:, dim]
            z_values.append((f"+{sweep_scale:g}std z{dim}", mu + offset))
            z_values.append((f"-{sweep_scale:g}std z{dim}", mu - offset))

        state_np = raw_state.numpy()
        _draw_tee(ax, goal_pose, color="tab:green", alpha=0.23, label="target T", linestyle="--")
        _draw_tee(ax, state_np[2:5], color="0.35", alpha=0.32, label="current T")
        ax.scatter([float(state_np[0])], [float(state_np[1])], c="black", s=38, zorder=6, label="pusher")
        chunks = []
        for label, z in z_values:
            z_emb = model.latent.embed(z)
            chunk = generate_chunk(model, obs_hist, act_hist, deterministic=True, z=z, z_emb=z_emb)["actions"]
            chunk_px = denormalize_actions_tensor(chunk.detach().cpu(), stats)[0] if stats is not None else chunk.detach().cpu()[0]
            chunks.append(chunk_px.numpy())
            linewidth = 2.2 if label == "mu" else 1.1
            alpha = 0.95 if label == "mu" else 0.55
            ax.plot(chunk_px[:, 0], chunk_px[:, 1], linewidth=linewidth, alpha=alpha, label=label)
        chunks_np = np.stack(chunks, axis=0)
        endpoint_spread = float(np.linalg.norm(chunks_np[:, -1] - chunks_np[:, -1].mean(axis=0, keepdims=True), axis=-1).mean())
        path_spread = float(np.linalg.norm(chunks_np - chunks_np.mean(axis=0, keepdims=True), axis=-1).mean())
        record = {
            "side": source["side"],
            "offset_from_goal_px": float(source["offset_from_goal_px"]),
            "sweep_scale": sweep_scale,
            "prior_mu": [float(x) for x in mu[0].detach().cpu()],
            "prior_std": [float(x) for x in std[0].detach().cpu()],
            "endpoint_spread_px": endpoint_spread,
            "path_spread_px": path_spread,
        }
        records.append(record)
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{source['side']} | offset={source['offset_from_goal_px']:.0f}px | endpoint spread={endpoint_spread:.2f}px")
        ax.legend(fontsize=6, loc="upper right", ncols=2)

    fig.savefig(output_dir / "latent_prior_direction_sweep.png", dpi=170)
    plt.close(fig)
    save_json(output_dir / "latent_prior_direction_sweep.json", {"records": records})
    endpoint_spreads = [item["endpoint_spread_px"] for item in records]
    path_spreads = [item["path_spread_px"] for item in records]
    return {
        "latent_sweep_endpoint_spread_mean": float(np.mean(endpoint_spreads)) if endpoint_spreads else 0.0,
        "latent_sweep_endpoint_spread_max": float(np.max(endpoint_spreads)) if endpoint_spreads else 0.0,
        "latent_sweep_path_spread_mean": float(np.mean(path_spreads)) if path_spreads else 0.0,
        "latent_sweep_num_states": float(len(records)),
        "latent_sweep_std_scale": float(sweep_scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n-exec", type=int, default=8)
    parser.add_argument("--render-episodes", type=int, default=4)
    parser.add_argument("--save-gifs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-samples", type=int, default=16)
    parser.add_argument("--latent-sample-panels", type=int, default=6)
    parser.add_argument("--contact-panels", type=int, default=6)
    parser.add_argument("--contact-grid-size", type=int, default=90)
    parser.add_argument("--wrong-side-samples", type=int, default=32)
    parser.add_argument("--latent-sweep-states", type=int, default=4)
    parser.add_argument("--latent-sweep-std-scale", type=float, default=2.0)
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated subset of variants to run. Use names like full_policy or group/name.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--skip-extra-diagnostics", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    raw = torch.load(checkpoint, map_location="cpu")
    base_config = apply_overrides(raw["config"], args.overrides)
    if args.device is not None:
        base_config.device = args.device

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(checkpoint, args.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(args.seed))
    device = resolve_device(str(base_config.get("device", "cpu")))
    model = build_model(base_config).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()
    variants = filter_variants(experiment_variants(int(args.n_exec)), args.variants)

    save_config(base_config, output_dir / "checkpoint_config.json")
    save_json(
        output_dir / "experiment_manifest.json",
        {
            "checkpoint": str(checkpoint),
            "output_dir": str(output_dir),
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "base_n_exec": int(args.n_exec),
            "render_episodes": int(args.render_episodes),
            "seed": int(args.seed),
            "num_workers": int(args.num_workers),
            "worker_threads": int(args.worker_threads),
            "skip_extra_diagnostics": bool(args.skip_extra_diagnostics),
            "variants": [{"group": item["group"], "name": item["name"], "description": item["description"]} for item in variants],
            "overrides": list(args.overrides),
        },
    )

    rows = []
    failures = []
    for idx, variant in enumerate(variants, start=1):
        variant_dir = output_dir / f"{variant['group']}__{variant['name']}"
        print(f"\n[{idx}] Running {variant['group']} / {variant['name']} -> {variant_dir}")
        config = make_eval_config(base_config, args, variant)
        save_config(config, variant_dir / "pusht_sim_config.json")
        save_json(variant_dir / "variant.json", variant)
        try:
            if int(args.num_workers) > 1:
                metrics = evaluate_pusht_sim_checkpoint_parallel(
                    checkpoint=checkpoint,
                    config=config,
                    device_name=str(config.get("device", "cpu")),
                    output_dir=variant_dir,
                    num_workers=int(args.num_workers),
                    worker_threads=int(args.worker_threads),
                )
            else:
                metrics = evaluate_pusht_sim_model(model, config, device, output_dir=variant_dir)
            row = {
                "group": variant["group"],
                "name": variant["name"],
                "description": variant["description"],
                "output_dir": str(variant_dir),
                **{key: value for key, value in metrics.items() if isinstance(value, (int, float, bool))},
            }
            rows.append(row)
            write_summary_csv(output_dir / "summary.csv", rows)
            save_json(output_dir / "summary.json", {"rows": rows, "failures": failures})
        except Exception as exc:
            failure = {
                "group": variant["group"],
                "name": variant["name"],
                "output_dir": str(variant_dir),
                "error": repr(exc),
            }
            failures.append(failure)
            save_json(output_dir / "summary.json", {"rows": rows, "failures": failures})
            if not args.continue_on_error:
                raise
            print(f"FAILED {variant['name']}: {exc!r}")

    if args.skip_extra_diagnostics:
        save_json(output_dir / "summary.json", {"rows": rows, "failures": failures})
        write_summary_csv(output_dir / "summary.csv", rows)
        print(f"\nCheckpoint-only experiments written to: {output_dir}")
        print(f"Completed variants: {len(rows)} | failures: {len(failures)}")
        if failures:
            print("Failures:")
            for failure in failures:
                print(f"  - {failure['group']} / {failure['name']}: {failure['error']}")
        return

    wrong_side_dir = output_dir / "latent_causal_use__wrong_side_go_around"
    print(f"\nRunning wrong-side go-around latent diagnostic -> {wrong_side_dir}")
    wrong_config = make_eval_config(
        base_config,
        args,
        {
            "group": "latent_causal_use",
            "name": "wrong_side_go_around",
            "description": "synthetic wrong-side latent sensitivity diagnostic",
            "overrides": {
                "eval.sim_n_exec": int(args.n_exec),
                "eval.sim_intervention": "full_policy",
                "eval.sim_latent_mode": "default",
            },
        },
    )
    try:
        wrong_metrics = plot_wrong_side_go_around_diagnostic(model, wrong_config, device, wrong_side_dir)
        save_json(wrong_side_dir / "wrong_side_metrics.json", wrong_metrics)
        rows.append(
            {
                "group": "latent_causal_use",
                "name": "wrong_side_go_around",
                "description": "synthetic wrong-side latent sensitivity diagnostic",
                "output_dir": str(wrong_side_dir),
                **wrong_metrics,
            }
        )
        write_summary_csv(output_dir / "summary.csv", rows)
        save_json(output_dir / "summary.json", {"rows": rows, "failures": failures})
    except Exception as exc:
        failure = {"group": "latent_causal_use", "name": "wrong_side_go_around", "output_dir": str(wrong_side_dir), "error": repr(exc)}
        failures.append(failure)
        save_json(output_dir / "summary.json", {"rows": rows, "failures": failures})
        if not args.continue_on_error:
            raise
        print(f"FAILED wrong_side_go_around: {exc!r}")

    sweep_dir = output_dir / "latent_causal_use__prior_direction_sweep"
    print(f"\nRunning prior-direction latent sweep -> {sweep_dir}")
    try:
        sweep_metrics = plot_prior_direction_latent_sweep(model, wrong_config, device, sweep_dir)
        save_json(sweep_dir / "latent_sweep_metrics.json", sweep_metrics)
        rows.append(
            {
                "group": "latent_causal_use",
                "name": "prior_direction_sweep",
                "description": "manual z sweep along prior mean +/- std coordinate directions",
                "output_dir": str(sweep_dir),
                **sweep_metrics,
            }
        )
        write_summary_csv(output_dir / "summary.csv", rows)
        save_json(output_dir / "summary.json", {"rows": rows, "failures": failures})
    except Exception as exc:
        failure = {"group": "latent_causal_use", "name": "prior_direction_sweep", "output_dir": str(sweep_dir), "error": repr(exc)}
        failures.append(failure)
        save_json(output_dir / "summary.json", {"rows": rows, "failures": failures})
        if not args.continue_on_error:
            raise
        print(f"FAILED prior_direction_sweep: {exc!r}")

    save_json(output_dir / "summary.json", {"rows": rows, "failures": failures})
    write_summary_csv(output_dir / "summary.csv", rows)
    print(f"\nCheckpoint-only experiments written to: {output_dir}")
    print(f"Completed variants: {len(rows)} | failures: {len(failures)}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure['group']} / {failure['name']}: {failure['error']}")


if __name__ == "__main__":
    main()
