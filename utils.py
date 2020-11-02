import os
import json
from argparse import Namespace


def get_config(path):
    with open(os.path.expanduser(path)) as f:
        config = json.loads(f.read(), object_hook=lambda d: Namespace(**d))

    return config
