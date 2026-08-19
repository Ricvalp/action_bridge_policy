"""Train query-only JAX policies on cached RLBench demonstrations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.training import train_state
from ml_collections import ConfigDict
from phi_rlbench.data.numpy_dataset import NumpyRLBenchDataset

from action_bridge.config import (
    apply_overrides,
    load_config,
    override_args,
    save_config,
    to_plain_dict,
)
from action_bridge.jax.eval.visualization import (
    prediction_chunk_figure,
    write_prediction_chunk_html,
)
from action_bridge.jax.models.config import (
    loss_config_from_config,
    policy_config_from_config,
)
from action_bridge.jax.models.rlbench_policy import (
    DirectChunkBCPolicy,
    RLBenchActionBridgePolicy,
)
from action_bridge.jax.training.checkpoints import load_checkpoint, save_checkpoint
from action_bridge.jax.training.data import (
    BackgroundBatchPrefetcher,
    BatchSource,
    dataset_kwargs,
)
from action_bridge.jax.training.losses import bridge_loss, direct_bc_loss
from action_bridge.jax.training.rlbench_online_metadata import (
    configure_rlbench_online_metadata,
)


class TrainState(train_state.TrainState):
    rng: jax.Array


def _build_dataset(config, split: str) -> NumpyRLBenchDataset:
    return NumpyRLBenchDataset(**dataset_kwargs(config, split))


def _build_model(config, dataset: NumpyRLBenchDataset):
    policy_config = policy_config_from_config(config)
    common = {
        "cfg": policy_config,
        "state_dim": int(dataset.state_dim),
        "action_dim": int(dataset.action_dim),
        "num_tasks": len(dataset.task_to_id),
        "num_task_variations": len(dataset.task_variation_to_id),
    }
    if str(config.policy_type) == "action_bridge":
        return RLBenchActionBridgePolicy(**common), policy_config
    if str(config.policy_type) == "direct_chunk_bc":
        return DirectChunkBCPolicy(**common), policy_config
    raise ValueError("policy_type must be action_bridge or direct_chunk_bc.")


def _parameter_count(params: Any) -> int:
    return int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(params)))


def _to_device(batch: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
    return jax.tree_util.tree_map(lambda value: jax.device_put(np.asarray(value)), batch)


def _training_devices(config) -> tuple[jax.Device, ...]:
    available = tuple(jax.local_devices())
    distributed = config.get("distributed", {})
    requested = int(distributed.get("num_devices", 0))
    if requested < 0:
        raise ValueError("distributed.num_devices must be non-negative.")
    count = len(available) if requested == 0 else requested
    if count < 1:
        raise RuntimeError("JAX did not expose any local devices.")
    if count > len(available):
        raise ValueError(
            f"distributed.num_devices={count} exceeds the {len(available)} "
            "devices visible to this process."
        )
    batch_size = int(config.optim.batch_size)
    if batch_size % count:
        raise ValueError(
            f"optim.batch_size={batch_size} must be divisible by "
            f"distributed.num_devices={count}."
        )
    return available[:count]


def _shard_batch(
    batch: Dict[str, np.ndarray], devices: tuple[jax.Device, ...]
) -> Dict[str, np.ndarray]:
    count = len(devices)

    def shard(value):
        array = np.asarray(value)
        if array.ndim < 1 or array.shape[0] % count:
            raise ValueError(
                f"Cannot shard batch leaf with shape {array.shape} over {count} devices."
            )
        per_device = array.shape[0] // count
        return array.reshape((count, per_device) + array.shape[1:])

    return jax.tree_util.tree_map(shard, batch)


def _merge_device_batch(tree: Any) -> Any:
    def merge(value):
        array = np.asarray(jax.device_get(value))
        if array.ndim < 2:
            return array
        return array.reshape((-1,) + array.shape[2:])

    return jax.tree_util.tree_map(merge, tree)


def _metric_value(value: Any) -> float:
    return float(np.mean(np.asarray(jax.device_get(value))))


def _state_step(state: TrainState, replicated: bool) -> int:
    step = jax_utils.unreplicate(state).step if replicated else state.step
    return int(np.asarray(jax.device_get(step)))


def _checkpoint_state(state: TrainState, replicated: bool) -> TrainState:
    return jax_utils.unreplicate(state) if replicated else state


def _create_steps(
    model,
    policy_config,
    loss_config,
    policy_type: str,
    *,
    devices: tuple[jax.Device, ...],
):
    is_bridge = str(policy_type) == "action_bridge"
    parallel = len(devices) > 1
    axis_name = "devices"

    def train_step(state: TrainState, batch: Dict[str, jax.Array]):
        next_rng, latent_rng, dropout_rng = jax.random.split(state.rng, 3)
        if parallel:
            device_index = jax.lax.axis_index(axis_name)
            latent_rng = jax.random.fold_in(latent_rng, device_index)
            dropout_rng = jax.random.fold_in(dropout_rng, device_index)

        def loss_fn(params):
            if is_bridge:
                output = model.apply(
                    {"params": params},
                    batch,
                    train=True,
                    use_posterior=True,
                    deterministic_latent=False,
                    rngs={"latent": latent_rng, "dropout": dropout_rng},
                )
                metrics = bridge_loss(
                    output,
                    batch,
                    loss_config,
                    policy_config.bridge,
                    state.step,
                )
            else:
                output = model.apply(
                    {"params": params}, batch, train=True, rngs={"dropout": dropout_rng}
                )
                metrics = direct_bc_loss(
                    output, batch, loss_config, policy_config.bridge
                )
            return metrics["loss"], metrics

        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        if parallel:
            gradients = jax.lax.pmean(gradients, axis_name=axis_name)
            metrics = jax.lax.pmean(metrics, axis_name=axis_name)
        metrics = {**metrics, "grad_norm": optax.global_norm(gradients)}
        state = state.apply_gradients(grads=gradients).replace(rng=next_rng)
        return state, metrics

    def validation_step(params, batch, step):
        if is_bridge:
            posterior_output = model.apply(
                {"params": params},
                batch,
                train=False,
                use_posterior=True,
                deterministic_latent=True,
            )
            posterior = bridge_loss(
                posterior_output,
                batch,
                loss_config,
                policy_config.bridge,
                step,
            )
            prior_output = model.apply(
                {"params": params},
                batch,
                train=False,
                use_posterior=False,
                deterministic_latent=True,
            )
            prior = bridge_loss(
                prior_output,
                batch,
                loss_config,
                policy_config.bridge,
                step,
            )
            metrics = {f"posterior_{key}": value for key, value in posterior.items()}
            metrics.update({f"prior_{key}": value for key, value in prior.items()})
            if parallel:
                metrics = jax.lax.pmean(metrics, axis_name=axis_name)
            return metrics, prior_output
        output = model.apply({"params": params}, batch, train=False)
        metrics = direct_bc_loss(
            output, batch, loss_config, policy_config.bridge
        )
        if parallel:
            metrics = jax.lax.pmean(metrics, axis_name=axis_name)
        return metrics, output

    if parallel:
        return (
            jax.pmap(train_step, axis_name=axis_name, devices=devices),
            jax.pmap(
                validation_step,
                axis_name=axis_name,
                in_axes=(0, 0, None),
                devices=devices,
            ),
        )
    return jax.jit(train_step), jax.jit(validation_step)


def _mean_metrics(metrics):
    return {
        key: float(np.mean([_metric_value(item[key]) for item in metrics]))
        for key in metrics[0]
    }


def _init_wandb(config, resume_payload):
    if not bool(config.logging.wandb.enabled):
        return None
    import wandb

    run_id = None
    if resume_payload is not None and bool(config.checkpoint.resume_wandb):
        run_id = resume_payload.get("wandb_run_id")
    return wandb.init(
        project=str(config.logging.wandb.project),
        entity=config.logging.wandb.entity,
        mode=str(config.logging.wandb.mode),
        name=str(config.run_id),
        group=config.logging.wandb.group,
        tags=list(config.logging.wandb.tags),
        config=to_plain_dict(config),
        id=run_id,
        resume="allow" if run_id else None,
    )


def _log(payload: Dict[str, float], step: int, wandb_run) -> None:
    line = {"step": int(step), **{key: round(value, 6) for key, value in payload.items()}}
    print(line, flush=True)
    if wandb_run is not None:
        wandb_run.log(payload, step=int(step))


def _validate(
    validation_step,
    params,
    source,
    batches: int,
    step: int,
    devices: tuple[jax.Device, ...],
):
    parallel = len(devices) > 1
    all_metrics = []
    last_batch = None
    last_output = None
    for _ in range(int(batches)):
        host_batch = source()
        last_batch = _shard_batch(host_batch, devices) if parallel else _to_device(host_batch)
        metrics, last_output = validation_step(
            params, last_batch, jnp.asarray(step, dtype=jnp.int32)
        )
        all_metrics.append(metrics)
    if parallel:
        last_batch = _merge_device_batch(last_batch)
        last_output = _merge_device_batch(last_output)
    return _mean_metrics(all_metrics), last_batch, last_output


def train(config) -> Path:
    if str(config.data.action_representation) != "absolute":
        raise ValueError(
            "The XYZ contact reference currently requires data.action_representation=absolute."
        )
    run_dir = Path(config.output_dir) / str(config.run_id)
    checkpoint_dir = run_dir / "checkpoints"
    figure_dir = run_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _build_dataset(config, "train")
    validation_dataset = _build_dataset(config, "val")
    resume_path = config.checkpoint.resume_path
    resume_payload = load_checkpoint(resume_path) if resume_path else None
    configure_rlbench_online_metadata(
        config,
        train_dataset,
        validation_dataset,
        resume_payload=resume_payload,
    )
    # Persist the resolved dataset vocabularies and cache identity, not only
    # the pre-instantiation user configuration.
    save_config(config, run_dir / "config.json")
    devices = _training_devices(config)
    parallel = len(devices) > 1
    model, policy_config = _build_model(config, train_dataset)
    loss_config = loss_config_from_config(config)
    initialization_source = BatchSource(
        train_dataset,
        batch_size=min(2, int(config.optim.batch_size)),
        sampling_strategy=str(config.data.sampling_strategy),
        seed=int(config.seed),
    )
    initialization_batch = _to_device(initialization_source())
    rng = jax.random.PRNGKey(int(config.seed))
    rng, parameter_rng, latent_rng, dropout_rng = jax.random.split(rng, 4)
    if str(config.policy_type) == "action_bridge":
        variables = model.init(
            {"params": parameter_rng, "latent": latent_rng, "dropout": dropout_rng},
            initialization_batch,
            train=True,
            use_posterior=True,
            deterministic_latent=False,
        )
    else:
        variables = model.init(
            {"params": parameter_rng, "dropout": dropout_rng},
            initialization_batch,
            train=True,
        )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.optim.grad_clip)),
        optax.adamw(
            learning_rate=float(config.optim.lr),
            weight_decay=float(config.optim.weight_decay),
        ),
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        rng=rng,
    )

    best_val_loss = float("inf")
    if resume_payload is not None:
        state = state.replace(
            step=jnp.asarray(resume_payload["step"], dtype=jnp.int32),
            params=resume_payload["params"],
            opt_state=resume_payload["opt_state"],
            rng=resume_payload["rng"],
        )
        best_val_loss = float(resume_payload.get("best_val_loss", best_val_loss))
        print(f"Resumed JAX checkpoint {resume_path} at step {int(state.step)}.", flush=True)

    wandb_run = _init_wandb(config, resume_payload)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "model_parameter_count": _parameter_count(state.params),
                "runtime_device_count": len(devices),
                "runtime_per_device_batch_size": int(config.optim.batch_size) // len(devices),
            },
            allow_val_change=True,
        )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "jax_visible_devices": [str(device) for device in jax.local_devices()],
                "training_devices": [str(device) for device in devices],
                "global_batch_size": int(config.optim.batch_size),
                "per_device_batch_size": int(config.optim.batch_size) // len(devices),
                "parameters": _parameter_count(state.params),
                "train_windows": len(train_dataset),
                "val_windows": len(validation_dataset),
                "tasks": len(train_dataset.task_to_id),
                "task_variations": len(train_dataset.task_variation_to_id),
            },
            indent=2,
        ),
        flush=True,
    )

    worker_datasets = []

    def make_source(worker_id: int):
        dataset = _build_dataset(config, "train")
        worker_datasets.append(dataset)
        return BatchSource(
            dataset,
            batch_size=int(config.optim.batch_size),
            sampling_strategy=str(config.data.sampling_strategy),
            seed=int(config.seed) + 1009 * (worker_id + 1),
        )

    prefetcher = BackgroundBatchPrefetcher(
        make_source,
        num_workers=int(config.data.prefetch_workers),
        max_prefetch=int(config.data.prefetch_batches),
    )
    validation_source = BatchSource(
        validation_dataset,
        batch_size=int(config.optim.batch_size),
        sampling_strategy=str(config.data.sampling_strategy),
        seed=int(config.seed) + 99173,
    )
    train_step, validation_step = _create_steps(
        model,
        policy_config,
        loss_config,
        str(config.policy_type),
        devices=devices,
    )
    if parallel:
        state = jax_utils.replicate(state, devices=devices)
    start_step = _state_step(state, parallel)
    start_time = time.monotonic()
    try:
        for target_step in range(start_step + 1, int(config.optim.max_steps) + 1):
            host_batch = prefetcher.get()
            batch = _shard_batch(host_batch, devices) if parallel else _to_device(host_batch)
            state, metrics = train_step(state, batch)
            if target_step == start_step + 1:
                jax.block_until_ready(state.params)
                print("JAX compilation complete.", flush=True)
            if target_step % int(config.logging.log_every_steps) == 0 or target_step == 1:
                elapsed = max(time.monotonic() - start_time, 1e-6)
                train_metrics = {
                    f"train/{key}": _metric_value(value)
                    for key, value in metrics.items()
                }
                train_metrics["train/steps_per_second"] = (target_step - start_step) / elapsed
                train_metrics["train/examples_per_second"] = (
                    (target_step - start_step) * int(config.optim.batch_size) / elapsed
                )
                train_metrics["data/prefetch_queue_size"] = float(prefetcher.qsize())
                _log(train_metrics, target_step, wandb_run)

            should_validate = target_step % int(config.logging.val_every_steps) == 0
            if should_validate or target_step == int(config.optim.max_steps):
                val_metrics, artifact_batch, artifact_output = _validate(
                    validation_step,
                    state.params,
                    validation_source,
                    int(config.logging.val_batches),
                    target_step,
                    devices,
                )
                val_metrics = {f"val/{key}": value for key, value in val_metrics.items()}
                _log(val_metrics, target_step, wandb_run)
                val_loss_key = (
                    "val/posterior_loss"
                    if str(config.policy_type) == "action_bridge"
                    else "val/loss"
                )
                if val_metrics[val_loss_key] < best_val_loss:
                    best_val_loss = val_metrics[val_loss_key]
                    if bool(config.checkpoint.get("save_best_val", True)):
                        save_checkpoint(
                            checkpoint_dir / "best_val.pt",
                            state=_checkpoint_state(state, parallel),
                            config=config,
                            best_val_loss=best_val_loss,
                            wandb_run_id=None if wandb_run is None else wandb_run.id,
                        )

                if target_step % int(config.logging.artifact_every_steps) == 0:
                    host_batch = jax.device_get(artifact_batch)
                    host_output = jax.device_get(artifact_output)
                    figure = prediction_chunk_figure(
                        host_batch,
                        host_output,
                        num_examples=int(config.logging.artifact_examples),
                    )
                    path = write_prediction_chunk_html(
                        figure,
                        figure_dir / f"step_{target_step:07d}" / "predicted_chunks.html",
                    )
                    print(f"Saved RLBench chunk diagnostics: {path}", flush=True)
                    if wandb_run is not None and bool(config.logging.wandb.log_plotly):
                        wandb_run.log({"val/predicted_chunks": figure}, step=target_step)

            if target_step % int(config.logging.checkpoint_every_steps) == 0:
                checkpoint_state = _checkpoint_state(state, parallel)
                save_checkpoint(
                    checkpoint_dir / f"step_{target_step:07d}.pt",
                    state=checkpoint_state,
                    config=config,
                    best_val_loss=best_val_loss,
                    wandb_run_id=None if wandb_run is None else wandb_run.id,
                )
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    state=checkpoint_state,
                    config=config,
                    best_val_loss=best_val_loss,
                    wandb_run_id=None if wandb_run is None else wandb_run.id,
                )
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            state=_checkpoint_state(state, parallel),
            config=config,
            best_val_loss=best_val_loss,
            wandb_run_id=None if wandb_run is None else wandb_run.id,
        )
    finally:
        prefetcher.close()
        train_dataset.close()
        validation_dataset.close()
        for dataset in worker_datasets:
            dataset.close()
        if wandb_run is not None:
            wandb_run.finish()
    return run_dir


def _parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="rlbench_jax_contact_bridge")
    arguments, unknown = parser.parse_known_args(argv)
    overrides = override_args(unknown)
    config = apply_overrides(load_config(arguments.config_name), overrides)
    resume_path = config.checkpoint.resume_path
    if resume_path:
        payload = load_checkpoint(resume_path)
        if payload.get("config"):
            restored = ConfigDict(payload["config"])
            restored.checkpoint.resume_path = str(resume_path)
            config = apply_overrides(restored, overrides)
    return config


def main(argv=None):
    train(_parse_config(argv))


if __name__ == "__main__":
    main()
