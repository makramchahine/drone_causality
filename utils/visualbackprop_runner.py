import json
import os.path
from typing import Dict, Tuple, Union, Optional

from utils.model_utils import NCPParams, LSTMParams, CTRNNParams, get_readable_name, TCNParams
from visual_backprop import run_visualbackprop


def visualbackprop_runner(datasets: Dict[str, Tuple[str, bool]], output_prefix: str = ".",
                          params_path: Optional[str] = None):
    """
    Convenience script that runs the run_visualbackprop function with
    the cross product of models and data paths and automatically generates output
    image and video names
    @param params_path:  model params are loaded from this json which maps checkpoint paths to model params repr() strs
    @param output_prefix: directory to put logs in
    @param models: dict mapping from model_type : model_path. Optionally contains
    @param datasets: dict mapping from dataset_name (for saving) : dataset path
    """
    with open(params_path, "r") as f:
        params_data = json.loads(f.read())

    for local_path, params_str in params_data.items():
        model_params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams, None] = eval(params_str)
        model_path = os.path.join(os.path.dirname(params_path), local_path)
        for dataset_name, (data_path, reverse_channels) in datasets.items():
            data_model_id = f"{get_readable_name(model_params)}_{dataset_name}"
            output_name = os.path.join(output_prefix, data_model_id)

            run_visualbackprop(
                model_path=model_path,
                data_path=data_path,
                model_params=model_params,
                image_output_path=output_name,
                video_output_path=os.path.join(output_name, f"{data_model_id}.mp4"),
                reverse_channels=reverse_channels,
            )
            print(f"Finished {data_model_id}")


if __name__ == "__main__":
    # models = {
    #     "ncp": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/ncp.hdf5",
    #     # "ncp_old": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/ncp_old.hdf5",
    #     "mixedcfc": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/mixedcfc.hdf5",
    #     "lstm": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/lstm.hdf5",
    # }
    #
    # datasets = {
    #     "online_ncp": ("/media/dolphonie/Data/Files/UROP/devens_data/11-18-21 online/raw_data/1637252567.166090", True),
    #     "online_cfc": ("/media/dolphonie/Data/Files/UROP/devens_data/11-18-21 online/raw_data/1637254105.961565", True),
    #     "online_lstm": (
    #         "/media/dolphonie/Data/Files/UROP/devens_data/11-18-21 online/raw_data/1637251162.043868", True),
    #     "winter": ("/media/dolphonie/Data/Files/UROP/devens_data/10-29-21 winter/1635515333.207994", True),
    #     "fall": ("/media/dolphonie/Data/Files/UROP/devens_data/8-4-21 fall/1628106140.64", False),
    # }
    # models = {
    #     "ncp2": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/online_1/rev-0_model-ncp_seq-64_opt-adam_lr-0.000273_crop-0.000000_epoch-086_val_loss:0.2399_mse:0.0689_2022:01:23:13:39:23.hdf5",
    #     "mixedcfc2": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/online_1/rev-0_model-ctrnn_ctt-mixedcfc_cn-1.000000_bba-silu_bb-dr-0.100000_fb-3.163336_bbu-253_bbl-1_wd-0.000000_mixed-0_seq-64_opt-adam_lr-0.000087_crop-0.000000_epoch-053_val-loss:0.2110_mse:0.0929_2022:01:24:07:54:15.hdf5",
    #     "lstm2": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/online_1/rev-0_model-lstm_seq-64_opt-adam_lr-0.000290_crop-0.000000_epoch-090_val_loss:0.1936_mse:0.0282_2022:01:22:03:19:54.hdf5",
    #     "cfc2": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/online_1/rev-0_model-ctrnn_ctt-cfc_cn-1.000000_bba-silu_bb-dr-0.100000_fb-3.009269_bbu-147_bbl-2_wd-0.000000_mixed-0_seq-64_opt-adam_lr-0.000183_crop-0.000000_epoch-097_val-loss:0.2078_mse:0.0530_2022:01:23:13:51:33.hdf5"
    # }

    datasets = {
        "snow_bag": (
            "/media/dolphonie/Data/Files/UROP/devens_data/1-26-22 online2/1643205757.184110", True),
        "snow_chair": (
            "/media/dolphonie/Data/Files/UROP/devens_data/1-26-22 online2/1643205847.856620", True),
    }

    # params_path = "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/online_1/params.json"
    params_path = "/home/dolphonie/Desktop/all_types_val/params.json"
    visualbackprop_runner(datasets, output_prefix="visualbackprop_results", params_path=params_path)
