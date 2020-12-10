import os
import json
from argparse import Namespace


def get_config(path):
    with open(os.path.expanduser(path)) as f:
        config = json.loads(f.read(), object_hook=lambda d: Namespace(**d))

    return config


def transform_label(target, pos_label):

    def transform_fn(a, v):
        if target == 'arousal':
            return int(a > 3) if pos_label == 'high' else int(a <= 3)
        else:
            return int(v > 2) if pos_label == 'high' else int(v <= 2)

    return transform_fn


def config_to_dict(config):
    if isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_to_dict(item) for item in config]
    elif isinstance(config, Namespace):
        return config_to_dict(vars(config))
    else:
        try:
            json.dumps(config)
            return config
        except (TypeError, OverflowError):
            return 'unserializable'
