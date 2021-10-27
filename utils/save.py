"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
from os.path import abspath, dirname, exists, join
import subprocess

import torch

from utils.logger import LOGGER


def save_training_meta(args):
    if args.rank > 0:
        return

    if not exists(args.output_dir):
        os.makedirs(join(args.output_dir, 'log'))
        os.makedirs(join(args.output_dir, 'ckpt'))

    with open(join(args.output_dir, 'log', 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    model_config = json.load(open(args.model_config))
    with open(join(args.output_dir, 'log', 'model.json'), 'w') as writer:
        json.dump(model_config, writer, indent=4)


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, optimizer=None):
        output_model_file = join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        torch.save(state_dict, output_model_file)
        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            if hasattr(optimizer, '_amp_stash'):
                pass
            torch.save(dump, f'{self.output_dir}/train_state_{step}.pt')
