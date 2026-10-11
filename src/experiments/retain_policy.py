"""Explicit data-access policy for the T0 general-only ablation (CPU safe)."""

POLICY = 'general-only-v1'
GROUPS = ('forget', 'retain', 'general')


def general_only(cfg):
    policy = cfg.get('selective', {}).get('training', {}).get('retain_policy')
    if policy is None:
        return False
    if policy != POLICY:
        raise ValueError('Unknown retain data-access policy: ' + str(policy))
    if cfg['unlearn']['batch_sizes'].get('retain') != 0:
        raise ValueError('General-only requires zero specified-retain batch')
    if any(cfg['conrep'][key] != 0 for key in ('specified_cl_weight', 'specified_lm_weight')):
        raise ValueError('General-only requires zero specified CL and LM weights')
    if cfg['baseline']['specified_weight'] != 0:
        raise ValueError('General-only requires zero baseline specified weight')
    if cfg['data'].get('retain_requests'):
        raise ValueError('General-only must not provide specified retain requests')
    return True


def active_groups(cfg):
    return ('forget', 'general') if general_only(cfg) else GROUPS


def sampling_seed(seed, group):
    # Removing a group must not change either F or G sampling streams.
    return seed + GROUPS.index(group) * 10007


def active_batch_config(cfg):
    return dict(cfg['unlearn'], batch_sizes={name: cfg['unlearn']['batch_sizes'][name]
                                           for name in active_groups(cfg)})
