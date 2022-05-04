# Created by Patrick Kao at 4/27/22
import json
import os
from pathlib import Path
from typing import Dict, Sequence, Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray


def graph_deltas_bad(means: Dict[str, ndarray], stds: Dict[str, ndarray], sorted_deltas: Sequence[float],
                     save_dir: Optional[str], display_result: bool = False):
    assert save_dir or display_result
    plt.clf()
    for model, model_distance in means.items():
        model_distance = np.asarray(model_distance)
        plt.plot(sorted_deltas, model_distance, label=model)
        std = np.asarray(stds[model])
        plt.fill_between(sorted_deltas, model_distance - std, model_distance + std, alpha=0.5)

    plt.legend(loc="upper left")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "distances.png"))
        with open(os.path.join(save_dir, "distances.json"), "w") as f:
            distances_stds = {"means": means, "stds": stds, "deltas": sorted_deltas}
            json.dump(distances_stds, f)

    if display_result:
        plt.show()


MARKERS = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']
LABEL_MAP = {
    "ncp": "NCP",
    "tcn": "TCN",
    "ctrnn_ctrnn": "CT-RNN",
    "lstm": "LSTM",
    "ctrnn_gruode": "GRUODE",
    "ctrnn_ltc": "LTC",
    "ctrnn_cfc": "CfC",
    "ctrnn_wiredcfccell": "Sparse-CfC",
    "ctrnn_mixedcfc": "Mixed-CfC"
}

GRAPH_ORDER = [
    'tcn', 'lstm', 'ctrnn_ctrnn', 'ctrnn_gruode', 'ctrnn_ltc', 'ncp', 'ctrnn_cfc', 'ctrnn_wiredcfccell'
]


def graph_deltas(means: Dict[str, ndarray], stds: Dict[str, ndarray], sorted_deltas: Sequence[float],
                 save_dir: Optional[str], display_result: bool = False, x_label: Optional[str] = None,
                 y_label: Optional[str] = None, force_even_x: bool = False):
    if set(GRAPH_ORDER) != set(means.keys()):
        print(f"Found different keys than graph order: {set(GRAPH_ORDER) - set(means.keys())}")

    plt.clf()
    # if force_even_x, don't use actual deltas, just use even range of numbers and override xtick labels
    x_axis = np.arange(len(sorted_deltas)) if force_even_x else np.array(sorted_deltas)
    normalized_x_axis = x_axis + (np.max(x_axis) * 0.09)

    # place a gray box between x= 25 and x = 45:
    plt.axvspan(normalized_x_axis[1] - ((normalized_x_axis[1] - normalized_x_axis[0]) / 2),
                normalized_x_axis[1] + ((normalized_x_axis[2] - normalized_x_axis[1]) / 2), facecolor='#cccccc',
                alpha=0.1)

    bar_marker = '|'

    for i, model_name in enumerate(GRAPH_ORDER):
        if model_name not in means:
            continue
        mean = np.asarray(means[model_name])
        std = np.asarray(stds[model_name])
        sns.scatterplot(x=x_axis + (np.max(x_axis) * i / 50), y=mean, s=1000 * std, alpha=0.3, label=None,
                        marker=bar_marker)
        # grab color of the last plot:
        color = plt.gca().collections[-1].get_facecolor()[0]
        sns.scatterplot(x=x_axis + (np.max(x_axis) * i / 50), y=mean, s=100, alpha=0.9,
                        label=LABEL_MAP[model_name], marker=MARKERS[i], color=color, )

    # remove top and right borders:
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # keep x-axis tick labels only for x_axis values; label with sorted_deltas even if x axis is overriden due to
    # force_even x to ensure it looks correct
    plt.xticks(ticks=x_axis + (np.max(x_axis) * 0.09), labels=sorted_deltas)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    # add horizontal grid lines:
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.4)

    # y-axis limits:
    y_limit = np.max([mean for mean in means.values()]) * 1.25
    plt.ylim(0, y_limit)

    # ticks inward:
    plt.tick_params(axis='both', which='both', direction='in', )

    # tick lengths y-axis:
    plt.tick_params(axis='both', which='both', length=0)

    # legend box off:
    # plt.legend(loc='upper left', frameon=False, fontsize=10)
    # place the legend horizontally:
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10)

    if save_dir:
        # save a high-res version of the figure in eps format:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig(os.path.join(save_dir, f"figure.pdf"), format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"figure.png"), format='png', dpi=600, bbox_inches='tight')
        with open(os.path.join(save_dir, "distances.json"), "w") as f:
            distances_stds = {"means": means, "stds": stds, "deltas": sorted_deltas}
            json.dump(distances_stds, f)

    if display_result:
        plt.show()
