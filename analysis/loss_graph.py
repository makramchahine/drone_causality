# Created by Patrick Kao at 3/29/22
import argparse
import os
import re
from typing import Optional, Sequence, Tuple, Dict

import matplotlib.pyplot as plt


def graph_loss(train_losses: Optional[Dict[str, Sequence[Tuple[int, float]]]] = None,
               val_losses: Optional[Dict[str, Sequence[Tuple[int, float]]]] = None, save_path: Optional[str] = None,
               display_result: bool = False):
    if train_losses:
        for name, loss in train_losses.items():
            plt.plot(*list(zip(*loss)), label=f"{name} Train Loss")
    if val_losses:
        for name, loss in val_losses.items():
            plt.plot(*list(zip(*loss)), label=f"{name} Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.savefig(save_path)

    if display_result:
        plt.show()


def get_losses_from_checkpoints(checkpoint_paths: Sequence[str], names=None):
    """
    Extracts epoch, train loss, val loss, tuples for all files that came from the same training run as
    checkpoint_path
    :param checkpoint_paths:
    :param names:
    :return:
    """
    all_train_losses = {}
    all_val_losses = {}
    for i, checkpoint_path in enumerate(checkpoint_paths):
        match_reg = re.compile(".*(\d\d\d\d:\d\d:\d\d:\d\d:\d\d:\d\d)")
        date_str = match_reg.search(checkpoint_path).group(1)
        dir_contents = os.listdir(os.path.dirname(checkpoint_path))
        same_run_dir = [path for path in dir_contents if date_str in path]
        train_losses = []
        val_losses = []
        for run in same_run_dir:
            epoch_reg = re.compile(".*epoch-(\d\d\d).*")
            train_reg = re.compile(".*train-loss:(\d*.\d\d\d\d)_.*")
            val_reg = re.compile(".*val-loss:(\d*.\d\d\d\d)_.*")
            epoch = int(epoch_reg.search(run).group(1))
            train_loss = float(train_reg.search(run).group(1))
            val_loss = float(val_reg.search(run).group(1))
            train_losses.append((epoch, train_loss))
            val_losses.append((epoch, val_loss))

        # sort both lists by epoch
        train_losses.sort(key=lambda x: x[0])
        val_losses.sort(key=lambda x: x[0])
        name = names[i] if names is not None else str(i)
        all_train_losses[name] = train_losses
        all_val_losses[name] = val_losses

    graph_loss(train_losses=all_train_losses, val_losses=all_val_losses, save_path="losses.png", display_result=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_paths", nargs='+', default=[])
    parser.add_argument('-n', '--names-list', nargs='+', default=None)
    args = parser.parse_args()
    get_losses_from_checkpoints(args.checkpoint_paths, args.names_list)
