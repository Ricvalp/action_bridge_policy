"""Plot recorded training trajectories versus EMA plans and short physics branches.

Example: python -m action_bridge.scripts.visualize_pusht_training_mismatch
  --checkpoint /path/to/latest.pt --dataset /path/to/pusht.zarr
Only load a checkpoint you trust. No training or backend changes are made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np
import torch

from action_bridge.data.pusht_adapter import PushTLowDimDataset
from action_bridge.data.revision_pusht import load_windows
from action_bridge.eval.revision_pusht import validate_evaluation_completion
from action_bridge.eval.revision_pusht_mismatch import reconstruct_mismatch, representative_examples
from action_bridge.eval.revision_pusht_mismatch_plots import plot_mismatch, plot_gap_summary
from action_bridge.plan_revision import checkpoints
from action_bridge.scripts.sb_pusht import dataset_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--episode', type=int, help='Use this dataset episode instead of randomly selecting episodes')
    parser.add_argument('--replans', type=int, default=16, help='Prefix length including startup')
    parser.add_argument('--seed', type=int, default=7301)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if min(args.episodes, args.threads) < 1 or args.replans < 2 or args.seed < 0:
        parser.error('Need positive episodes/threads, replans >= 2, and seed >= 0')
    torch.set_num_threads(args.threads)
    with args.checkpoint.open('rb') as stream:
        state = checkpoints.load(stream)
        stream.seek(0)
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    config, metadata = state['config'], state['metadata']
    if config.get('protocol') not in ('self_source_v1', 'self_source_k_curriculum_v1') or config['method'] == 'ddim':
        parser.error('Use an FM/SB self-source reviser checkpoint')
    if config['method'].startswith('sb_') and state.get('direction') != 'forward':
        parser.error('Use a forward-phase SB policy checkpoint')
    mode = config.get('completion_id', 2)
    validate_evaluation_completion(config, mode)
    if (config['obs_dim'], config['action_dim']) != (5, 2) or not 0 < config['execute'] < config['horizon']:
        parser.error('Requires Push-T state/xy targets and 0 < K < H')
    if dataset_digest(args.dataset) != metadata.get('dataset_sha256'):
        parser.error('Dataset content hash does not match the checkpoint')
    print('Loading matching demonstration windows; CPU only.', flush=True)
    windows, current_metadata = load_windows(args.dataset, config)
    for key in ('normalization', 'codec', 'splits', 'normalizer_id', 'observation_profile', 'action_profile', 'train_subset'):
        if current_metadata.get(key) != metadata.get(key):
            parser.error(f'Dataset reconstruction differs from checkpoint: {key}')
    raw = PushTLowDimDataset(str(args.dataset), split='all', normalize=False, pad_episode_starts=True,
                            obs_key='data/state' if args.dataset.suffix == '.zarr' else None,
                            chunk_horizon=config['horizon'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir or Path(__file__).resolve().parents[2] / 'workspace' / 'sb_pusht' / 'training_mismatch' / stamp
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    print(f'Reconstructing EMA replay on {args.split} histories, then branching physics.', flush=True)
    result = reconstruct_mismatch(
        checkpoints.restore_policy(state, 'cpu'), config, metadata, state['dependencies'],
        windows[args.split], raw.observations, raw.actions,
        episodes=args.episodes, replans=args.replans, seed=args.seed, episode=args.episode,
        completion_id=mode, progress=args.progress)
    for record in result['records']:
        record['split'] = args.split
    examples = []
    for label, record in representative_examples(result['records']):
        name = f'{label}-episode{record["episode_id"]}-step{record["step"]}.png'
        plot_mismatch(record, output / name)
        examples.append(dict(selection=label, episode_id=record['episode_id'], step=record['step'], file=name))
    plot_gap_summary(result['records'], output / 'gap_summary.png')
    summary = {key: dict(mean=float(np.mean(values)), median=float(np.median(values)),
                         p90=float(np.quantile(values, .9)))
               for key in result['records'][0]['metrics']
               for values in [[r['metrics'][key] for r in result['records']]]}
    result.update(summary=summary, examples=examples, identity=dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        checkpoint_step=state['step'], weights='ema', dataset=str(args.dataset.resolve()),
        dataset_sha256=metadata['dataset_sha256'], split=args.split, config=config,
        metadata=metadata, checkpoint_runtime=state.get('runtime'), runtime=checkpoints.runtime_identity(),
        versions={p: version(p) for p in ('torch', 'gym-pusht', 'pymunk', 'numpy', 'matplotlib')},
        device='cpu', threads=args.threads,
        interpretation='Reconstructed all-self replay, not the original frozen block cache. '
                       'Each physics branch starts at the demonstration pose with reset hidden state; '
                       'it is not an exact simulator clone or a continuous closed-loop rollout. '
                       'The expert-action branch measures replay/reset mismatch. '
                       'Actual offline training continues to use recorded states and recorded action histories.'))
    (output / 'diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    assert checkpoints.digest(args.checkpoint) == checkpoint_hash
    print(f'Saved {len(result["records"])} comparisons; skipped {len(result["skipped"])} early terminations: {output}', flush=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
