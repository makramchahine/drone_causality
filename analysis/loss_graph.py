# Created by Patrick Kao at 3/29/22
import argparse
import os
import re
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def graph_loss(train_losses: Optional[Sequence[Tuple[int, float]]] = None,
               val_losses: Optional[Sequence[Tuple[int, float]]] = None, save_path: Optional[str] = None,
               display_result: bool = False):
    if train_losses:
        plt.plot(*list(zip(*train_losses)), label="Train Loss")
    if val_losses:
        plt.plot(*list(zip(*val_losses)), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.savefig(save_path)

    if display_result:
        plt.show()


def get_losses_from_checkpoints(checkpoint_path: str):
    """
    Extracts epoch, train loss, val loss, tuples for all files that came from the same training run as
    checkpoint_path
    :param checkpoint_path:
    :return:
    """
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
    graph_loss(train_losses=train_losses, val_losses=val_losses, save_path="losses.png", display_result=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    args = parser.parse_args()
    get_losses_from_checkpoints(args.checkpoint_path)
