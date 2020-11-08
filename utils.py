import os
import json
from argparse import Namespace


def get_config(path):
    with open(os.path.expanduser(path)) as f:
        config = json.loads(f.read(), object_hook=lambda d: Namespace(**d))

    return config


def transform_label(target, pos_label):

    def transform_fn(a, v):
        label = a if target == 'arousal' else v
        return int(label > 2) if pos_label == 'high' else int(label <= 2)

    return transform_fn
