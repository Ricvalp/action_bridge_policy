"""Compare recorded training histories with short, same-pose physics branches.

This reconstructs offline self-source replay with a supplied frozen EMA. It is
not the original historical training cache and not an on-policy rollout: each
K-step counterfactual starts again at the corresponding demonstration pose.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from action_bridge.eval.revision_pusht_mismatch_sim import simulate_commands
from action_bridge.plan_revision.completion import complete_plan
from action_bridge.plan_revision.contracts import ActionCodec, take
from action_bridge.plan_revision.curriculum import is_curriculum
from action_bridge.plan_revision.data import build_self_sources, immutable_snapshot
from action_bridge.plan_revision.tracking import preserve_rng
from action_bridge.plan_revision.training import restore_completion


def select_prefixes(windows, config, *, episodes, replans, seed, episode=None):
    """Keep ancestors from t=0; never generate an old plan in isolation."""
    ids = windows['episode_id'].unique(sorted=True)
    if episode is None:
        order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
        selected = ids[order[:episodes]].tolist()
    else:
        if episode not in ids.tolist():
            raise ValueError(f'Episode {episode} is not in the selected dataset split')
        selected = [episode]
    indices = []
    for identifier in selected:
        rows = (windows['episode_id'] == identifier).nonzero().flatten()
        rows = rows[windows['time_index'][rows].argsort()]
        last_step = (replans - 1) * config['execute']
        rows = rows[windows['time_index'][rows] <= last_step]
        spacing = 1 if is_curriculum(config) else config['execute']
        times = windows['time_index'][rows]
        if not len(times) or times[0] != 0 or bool((times.diff() != spacing).any()):
            raise ValueError('Replay prefixes must start at zero with no missing decisions')
        indices.extend(rows.tolist())
    return take(windows, torch.tensor(indices, dtype=torch.long)), selected


def compare_paths(demo_states, demo_commands, old_plan, policy_states, expert_states, execute):
    """Distances in raw pixels; command targets are not physical positions."""
    demo, policy, expert = (np.asarray(path) for path in (demo_states, policy_states, expert_states))
    old, commands = np.asarray(old_plan), np.asarray(demo_commands)
    if any(path.shape != (execute + 1, 5) for path in (demo, policy, expert)):
        raise ValueError('Compare complete, aligned K+1 state trajectories only')
    distance = lambda a, b: float(np.linalg.norm(a - b))
    return dict(
        pusher_counterfactual_gap_px=distance(policy[-1, :2], demo[-1, :2]),
        expert_replay_gap_px=distance(expert[-1, :2], demo[-1, :2]),
        simulated_action_effect_px=distance(policy[-1, :2], expert[-1, :2]),
        retained_to_demo_pusher_gap_px=distance(old[execute], demo[-1, :2]),
        retained_to_simulated_pusher_gap_px=distance(old[execute], policy[-1, :2]),
        previous_command_mismatch_px=distance(old[execute - 1], commands[-1]),
        block_counterfactual_gap_px=distance(policy[-1, 2:4], demo[-1, 2:4]),
        expert_block_replay_gap_px=distance(expert[-1, 2:4], demo[-1, 2:4]),
    )


@torch.no_grad()
def reconstruct_mismatch(policy, config, metadata, dependencies, windows, observations, actions,
                         *, device='cpu', episodes=16, replans=16, seed=7301, episode=None,
                         completion_id=2, progress=True, simulate=simulate_commands):
    """Replay EMA plans on data, then execute their prefixes in separate physics branches."""
    k = config['execute']
    if not 0 < k < config['horizon']:
        raise ValueError('A retained-plan diagnostic requires 0 < K < H')
    replay_config = dict(config)
    if is_curriculum(config):
        # Diagnose deployment K, not the first block of the training curriculum.
        replay_config['k_values_by_block'] = [k] * config['training_blocks']
    with preserve_rng():
        prefix, selected = select_prefixes(windows, config, episodes=episodes,
                                           replans=replans, seed=seed, episode=episode)
        completion = restore_completion(dependencies, device)
        replay, diagnostics = build_self_sources(
            prefix, immutable_snapshot(policy).to(device), completion,
            torch.as_tensor(dependencies['innovation_variance']), replay_config, device,
            block=0, seed=seed, p_self=1., modes=(completion_id,))
        codec = ActionCodec(**metadata['codec'])
        lookup = {(int(e), int(t)): i for i, (e, t) in enumerate(
            zip(replay['episode_id'], replay['time_index']))}
        rows, skipped = [], []
        indices = replay['has_previous_plan'].nonzero().flatten().tolist()
        for index in tqdm(indices, desc='Comparing data / EMA physics', disable=not progress):
            row = take(replay, index, device)
            identifier, step = int(row['episode_id']), int(row['time_index'])
            start = step - k
            previous = lookup[identifier, start]
            # The retained plan must really descend from the previous EMA query.
            torch.testing.assert_close(row['old_actions'].cpu(), replay['generated_actions'][previous], rtol=0, atol=0)
            demo = np.asarray(observations[identifier][start:step + 1], dtype=np.float32)
            commands = np.asarray(actions[identifier][start:step], dtype=np.float32)
            state = row['obs_hist'][-1].cpu().numpy()
            stats = metadata['normalization']
            state = state * np.asarray(stats['obs_std']) + np.asarray(stats['obs_mean'])
            np.testing.assert_allclose(state, demo[-1], atol=1e-4, rtol=1e-6)
            old = codec.decode(row['old_actions']).cpu().numpy()
            np.testing.assert_allclose(codec.decode(row['act_hist'][-1]).cpu(), commands[-1], atol=1e-4)
            # Both branches have the same reconstructed pose and hidden-state assumption.
            simulation_seed = seed + identifier * 10000 + start
            simulated_policy = simulate(demo[0], old[:k], seed=simulation_seed)
            simulated_expert = simulate(demo[0], commands, seed=simulation_seed)
            if any(len(branch['states']) != k + 1 for branch in (simulated_policy, simulated_expert)):
                skipped.append(dict(episode_id=identifier, step=step, reason='terminated before K actions',
                                    policy_steps=len(simulated_policy['states']) - 1,
                                    expert_steps=len(simulated_expert['states']) - 1))
                continue
            np.testing.assert_allclose(simulated_policy['states'][0], simulated_expert['states'][0], atol=1e-5)
            completed = complete_plan(
                row['old_actions'][None], k, row['obs_hist'][None], row['act_hist'][None],
                completion_id, completion, robot_dt=config['robot_dt'],
                velocity_weighting=config.get('completion_velocity_weighting', 'last_pair'),
                allow_full_completion=is_curriculum(config))
            rows.append(dict(
                episode_id=identifier, start_step=start, step=step, execute=k,
                demo_states=demo.tolist(), demo_commands=commands.tolist(), old_plan=old.tolist(),
                completed_source=codec.decode(completed)[0].cpu().tolist(),
                noisy_source=codec.decode(row['source_actions']).cpu().tolist(),
                current_target=codec.decode(row['future_actions']).cpu().tolist(),
                current_prediction=codec.decode(row['generated_actions']).cpu().tolist(),
                simulated_policy=simulated_policy, simulated_expert=simulated_expert,
                metrics=compare_paths(demo, commands, old, simulated_policy['states'],
                                      simulated_expert['states'], k)))
    if not rows:
        raise ValueError('No complete non-startup K-step comparisons available')
    return dict(records=rows, skipped=skipped, replay=diagnostics,
                selection=dict(episode_ids=selected, replans=replans, seed=seed,
                               source_probability=1., execute=k, reconstructed_not_historical=True))


def representative_examples(records):
    """Label selection honestly: typical, high gap, and a well-controlled example."""
    ordered = sorted(records, key=lambda r: r['metrics']['pusher_counterfactual_gap_px'])
    chosen = [('median_gap', ordered[round(.5 * (len(ordered) - 1))]),
              ('p90_gap', ordered[round(.9 * (len(ordered) - 1))])]
    controlled = [r for r in records if r['metrics']['expert_replay_gap_px'] <= 5.]
    if controlled:
        chosen.append(('large_action_effect_expert_control_under_5px',
                       max(controlled, key=lambda r: r['metrics']['simulated_action_effect_px'])))
    return chosen
