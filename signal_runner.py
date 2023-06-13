import argparse
import copy
import json
import os.path
from enum import Enum
from typing import Dict, Tuple, Union, Optional, Any

from utils.signal_utils import run_signal, parse_params_json
from hyperparameter_tuning import parse_unknown_args
from utils.model_utils import NCPParams, LSTMParams, CTRNNParams, get_readable_name, TCNParams, ModelParams, \
    load_model_from_weights

def get_sig_models(model_path: str, model_params: ModelParams, sig_kwargs: Dict[str, Any]):
    sig_params = copy.deepcopy(model_params)
    # model params already has single step true, set again for redundancy
    sig_params.single_step = True
    sig_params.no_norm_layer = False
    sig_model = load_model_from_weights(sig_params, model_path)

    return sig_model


def signal_each(datasets: Dict[str, Tuple[str, bool]], output_prefix: str = ".",
                   params_path: Optional[str] = None, include_checkpoint_name: bool = False,
                   sig_model_type: Optional[str] = None,
                   match_net_type: bool = False,
                   **sig_kwargs):
    """
    Convenience script that runs the run_signal function with
    the cross product of models and data paths and automatically generates output
    image and video names
    @param params_path:  model params are loaded from this json which maps checkpoint paths to model params repr() strs
    @param output_prefix: directory to put logs in
    @param models: dict mapping from model_type : model_path. Optionally contains
    @param datasets: dict mapping from dataset_name (for saving) : dataset path
    """
    # args only for compatibility with command line runner
    if len(sig_kwargs):
        print(f"Not using args {sig_kwargs}")

    if sig_kwargs is None:
        sig_kwargs = {}

    for local_path, model_path, model_params in parse_params_json(params_path):

        net_name = get_readable_name(model_params)
        for dataset_name, (data_path, reverse_channels, csv_path) in datasets.items():
            checkpoint_name = f"_{os.path.splitext(local_path)[0]}" if include_checkpoint_name else ""
            data_model_id = f"{get_readable_name(model_params)}_{dataset_name}{checkpoint_name}"
            output_name = os.path.join(output_prefix, data_model_id)

            # skip if explicitly only one sig type to be done
            if sig_model_type is not None and net_name != sig_model_type:
                continue

            if match_net_type and net_name not in data_path:
                continue

            sig_model = get_sig_models(model_path, model_params, sig_kwargs)
            run_signal(
                sig_model=sig_model,
                data=data_path,
                output_path=os.path.join(output_name, f"{data_model_id}."),###### METS LE TYPE DE LA BDD
                reverse_channels=reverse_channels,
                sig_kwargs=sig_kwargs,
            )
            print(f"Finished {data_model_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("params_path", type=str)
    parser.add_argument("--include_checkpoint_name", action="store_true")
    parser.add_argument("--sig_model_type", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="signal_processing_results")
    parser.add_argument("--match_net_type", action="store_true")
    args, unknown_args = parser.parse_known_args()
    arg_sig_kwargs = parse_unknown_args(unknown_args)

    with open(args.dataset_path, "r") as f:
        datasets = json.load(f)

    signal_each(datasets=datasets, output_prefix=args.output_prefix, params_path=args.params_path,
             include_checkpoint_name=args.include_checkpoint_name,
             sig_model_type=args.sig_model_type, sig_kwargs=arg_sig_kwargs, match_net_type=args.match_net_type)