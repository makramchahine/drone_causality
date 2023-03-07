# Created by Patrick Kao at 3/29/22
import argparse
import os
import re
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import seaborn as sns

# noinspection PyArgumentList
from utils.graph_utils import MARKERS


# noinspection PyArgumentList
class GraphMode(Enum):
    TRAIN = auto()
    VAL = auto()
    BOTH = auto()


def graph_loss(train_losses: Optional[Dict[str, Sequence[Tuple[int, float]]]] = None,
               val_losses: Optional[Dict[str, Sequence[Tuple[int, float]]]] = None, save_dir: Optional[str] = None,
               display_result: bool = False, marker_interval: int = 20):
    plt.clf()

    marker_count = 0
    if train_losses:
        for name, loss in train_losses.items():
            loss_transposed = list(zip(*loss))
            sns.lineplot(x=loss_transposed[0], y=loss_transposed[1], label=f"{name} Train Loss", ci=None,
                         marker=MARKERS[marker_count], markevery=marker_interval, markersize=8)
            marker_count += 1
    if val_losses:
        for name, loss in val_losses.items():
            loss_transposed = list(zip(*loss))
            sns.lineplot(x=loss_transposed[0], y=loss_transposed[1], label=f"{name} Validation Loss", ci=None,
                         marker=MARKERS[marker_count], markevery=marker_interval, markersize=8)
            marker_count += 1

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.autoscale(enable=True, axis='x', tight=True)

    # remove top and right borders:
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # ticks inward:
    plt.tick_params(axis='both', which='both', direction='in', )

    # tick lengths y-axis:
    plt.tick_params(axis='both', which='both', length=0)

    # legend box off:
    # plt.legend(loc='upper left', frameon=False, fontsize=10)
    # place the legend horizontally:
    plt.legend(loc='upper right', frameon=False, fontsize=10)

    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig(os.path.join(save_dir, f"figure.pdf"), format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"figure.png"), format='png', dpi=600, bbox_inches='tight')

    if display_result:
        plt.show()


def get_losses_from_checkpoints(checkpoint_paths: Sequence[str], save_dir: str, names=None,
                                mode: GraphMode = GraphMode.BOTH):
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

        name_search = re.compile("model-(.*)_seq-.*")
        model_name = name_search.search(checkpoint_path).group(1)

        dir_contents = os.listdir(os.path.dirname(checkpoint_path))
        same_run_dir = [path for path in dir_contents if (date_str in path and model_name in path)]
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

    train_graph = all_train_losses if mode != GraphMode.VAL else None
    val_graph = all_val_losses if mode != GraphMode.TRAIN else None
    graph_loss(train_losses=train_graph, val_losses=val_graph, save_dir=save_dir, display_result=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_paths", nargs='+', default=[])
    parser.add_argument('-n', '--names-list', nargs='+', default=None)
    parser.add_argument("--mode", default="both")
    parser.add_argument("--save_dir", default="loss_out")
    args = parser.parse_args()

    mode = GraphMode[args.mode.upper()]
    get_losses_from_checkpoints(args.checkpoint_paths, args.save_dir, args.names_list, mode)
