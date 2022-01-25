import os.path
from typing import Dict, Tuple

from visual_backprop import run_visualbackprop


def visualbackprop_runner(models: Dict[str, str], datasets: Dict[str, Tuple[str, bool]], output_prefix: str = "."):
    """
    Convenience script that runs the run_visualbackprop function with
    the cross product of models and data paths and automatically generates output
    image and video names
    @param output_prefix: directory to put logs in
    @param models: dict mapping from model_type : model_path
    @param datasets: dict mapping from dataset_name (for saving) : dataset path
    """
    for model_type, model_path in models.items():
        for dataset_name, (data_path, reverse_channels) in datasets.items():
            data_model_id = f"{model_type}_{dataset_name}"
            output_name = os.path.join(output_prefix, data_model_id)
            run_visualbackprop(
                model_path=model_path,
                model_type=model_type,
                data_path=data_path,
                image_output_path=output_name,
                video_output_path=os.path.join(output_name, f"{data_model_id}.mp4"),
                reverse_channels=reverse_channels,
            )
            print(f"Finished {data_model_id}")


if __name__ == "__main__":
    models = {
        "ncp": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/ncp.hdf5",
        # "ncp_old": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/ncp_old.hdf5",
        "mixedcfc": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/mixedcfc.hdf5",
        "lstm": "/home/dolphonie/projects/catkin_ws/src/rosetta_drone/rnn_control/src/models/lstm.hdf5",
    }

    datasets = {
        "online_ncp": ("/media/dolphonie/Data/Files/UROP/devens_data/11-18-21 online/raw_data/1637252567.166090", True),
        "online_cfc": ("/media/dolphonie/Data/Files/UROP/devens_data/11-18-21 online/raw_data/1637254105.961565", True),
        "online_lstm": (
            "/media/dolphonie/Data/Files/UROP/devens_data/11-18-21 online/raw_data/1637251162.043868", True),
        "winter": ("/media/dolphonie/Data/Files/UROP/devens_data/10-29-21 winter/1635515333.207994", True),
        "fall": ("/media/dolphonie/Data/Files/UROP/devens_data/8-4-21 fall/1628106140.64", False),
    }
    visualbackprop_runner(models, datasets, output_prefix="visualbackprop_results")
