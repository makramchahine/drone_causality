import argparse
import copy
import json
import os.path
from enum import Enum
from typing import Dict, Tuple, Union, Optional, Any

from analysis.grad_cam import get_last_conv, compute_gradcam, compute_gradcam_tile
from analysis.input_grad import compute_input_grad
from analysis.shap_heatmap import compute_shap
from utils.vis_utils import run_visualization, write_video, parse_params_json
from analysis.visual_backprop import get_conv_head, compute_visualbackprop
from hyperparameter_tuning import parse_unknown_args
from utils.model_utils import NCPParams, LSTMParams, CTRNNParams, get_readable_name, TCNParams, ModelParams, \
    load_model_from_weights


class VisualizationType(Enum):
    VISUAL_BACKPROP = "visual_backprop"
    GRAD_CAM = "grad_cam"
    GRAD_CAM_TILE = "grad_cam_tile"
    INPUT_GRAD = "input_grad"
    SHAP = "shap"


def get_vis_models(vis_type: VisualizationType, model_path: str, model_params: ModelParams, vis_kwargs: Dict[str, Any]):
    if vis_type == VisualizationType.VISUAL_BACKPROP:
        vis_model = get_conv_head(model_path, model_params)
        vis_func = compute_visualbackprop
    elif vis_type == VisualizationType.GRAD_CAM:
        vis_model = get_last_conv(model_path, model_params)
        vis_func = compute_gradcam
    elif vis_type == VisualizationType.GRAD_CAM_TILE:
        vis_model = get_last_conv(model_path, model_params)
        vis_func = compute_gradcam_tile
    elif vis_type == VisualizationType.INPUT_GRAD:
        vis_model = load_model_from_weights(model_params, model_path)
        vis_func = compute_input_grad
    elif vis_type == VisualizationType.SHAP:
        vis_model = load_model_from_weights(model_params, model_path)
        vis_func = compute_shap
        assert "cache_path" in vis_kwargs
        assert "dataset_path" in vis_kwargs
    else:
        raise ValueError("Illegal vis type")

    control_params = copy.deepcopy(model_params)
    # model params already has single step true, set again for redundancy
    control_params.single_step = True
    control_params.no_norm_layer = False
    control_model = load_model_from_weights(control_params, model_path)

    return vis_model, vis_func, control_model


def visualize_each(datasets: Dict[str, Tuple[str, bool]], output_prefix: str = ".",
                   params_path: Optional[str] = None, include_checkpoint_name: bool = False,
                   vis_type: VisualizationType = VisualizationType.VISUAL_BACKPROP,
                   vis_model_type: Optional[str] = None, vis_kwargs: Optional[Dict[str, Any]] = None,
                   match_net_type: bool = False, absolute_norm: bool = True,
                   **kwargs):
    """
    Convenience script that runs the run_visualbackprop function with
    the cross product of models and data paths and automatically generates output
    image and video names
    @param params_path:  model params are loaded from this json which maps checkpoint paths to model params repr() strs
    @param output_prefix: directory to put logs in
    @param models: dict mapping from model_type : model_path. Optionally contains
    @param datasets: dict mapping from dataset_name (for saving) : dataset path
    """
    # args only for compatibility with command line runner
    if len(kwargs):
        print(f"Not using args {kwargs}")

    assert not (vis_model_type and match_net_type), "Only one of vis_model_type and match_net_type should be specified"

    if vis_kwargs is None:
        vis_kwargs = {}

    for local_path, model_path, model_params in parse_params_json(params_path):
        net_name = get_readable_name(model_params)
        for dataset_name, (data_path, reverse_channels, csv_path) in datasets.items():
            checkpoint_name = f"_{os.path.splitext(local_path)[0]}" if include_checkpoint_name else ""
            data_model_id = f"{get_readable_name(model_params)}_{dataset_name}{checkpoint_name}"
            output_name = os.path.join(output_prefix, data_model_id)

            # skip if explicitly only one vis type to be done
            if vis_model_type is not None and net_name != vis_model_type:
                continue

            if match_net_type and net_name not in data_path:
                continue

            vis_model, vis_func, control_model = get_vis_models(vis_type, model_path, model_params, vis_kwargs)
            run_visualization(
                vis_model=vis_model,
                data=data_path,
                vis_func=vis_func,
                image_output_path=None,
                video_output_path=os.path.join(output_name, f"{data_model_id}.mp4"),
                reverse_channels=reverse_channels,
                control_source=csv_path if csv_path else control_model,
                vis_kwargs=vis_kwargs,
                absolute_norm=absolute_norm,
            )
            print(f"Finished {data_model_id}")


def visualize_combined(datasets: Dict[str, Tuple[str, bool]], output_prefix: str = ".",
                       params_path: Optional[str] = None,
                       vis_type: VisualizationType = VisualizationType.VISUAL_BACKPROP,
                       num_keep_frames: Optional[int] = None, control_csv: Optional[str] = None,
                       absolute_norm: bool = True,
                       **kwargs):
    """
    Script that instead of producing one output video per dataset per model, combines all of the videos from all of the
    models in params_path

    :param datasets:
    :param output_prefix:
    :param params_path:
    :param include_checkpoint_name:
    :param vis_type:
    :return:
    """
    # args only for compatibility with command line runner
    if len(kwargs):
        print(f"Not using args {kwargs}")

    with open(params_path, "r") as f:
        params_data = json.loads(f.read())

    for dataset_name, (data_path, reverse_channels) in datasets.items():
        img_frames = []
        for local_path, params_str in params_data.items():
            model_params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams, None] = eval(params_str)
            model_params.single_step = True
            model_path = os.path.join(os.path.dirname(params_path), local_path)

            vis_model, vis_func, control_model = get_vis_models(vis_type, model_path, model_params)
            imgs = run_visualization(
                vis_model=vis_model,
                data=data_path,
                vis_func=vis_func,
                image_output_path=None,
                video_output_path=None,
                reverse_channels=reverse_channels,
                control_source=control_csv if control_csv is not None else control_model,
                absolute_norm=absolute_norm,
            )
            last_kept_frame = 0 if num_keep_frames is None else len(imgs) - num_keep_frames
            img_frames.extend(imgs[last_kept_frame:])

        write_video(img_frames, os.path.join(output_prefix, f"{dataset_name}.mp4"))
        print(f"Finished {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vis_func", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("params_path", type=str)
    parser.add_argument("--vis_type", type=str, default=VisualizationType.VISUAL_BACKPROP.value)
    parser.add_argument("--include_checkpoint_name", action="store_true")
    parser.add_argument("--num_keep_frames", type=int, default=None)
    parser.add_argument("--vis_model_type", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="visualbackprop_results")
    parser.add_argument("--match_net_type", action="store_true")
    parser.add_argument('--absolute_norm', action='store_true')
    parser.add_argument('--no_absolute_norm', dest="absolute_norm", action='store_false')
    parser.set_defaults(absolute_norm=True)
    args, unknown_args = parser.parse_known_args()
    arg_vis_kwargs = parse_unknown_args(unknown_args)

    vis_func = locals()[args.vis_func.lower()]

    with open(args.dataset_path, "r") as f:
        datasets = json.load(f)

    vis_func(datasets=datasets, output_prefix=args.output_prefix, params_path=args.params_path,
             vis_type=VisualizationType(args.vis_type),
             include_checkpoint_name=args.include_checkpoint_name, num_keep_frames=args.num_keep_frames,
             vis_model_type=args.vis_model_type, vis_kwargs=arg_vis_kwargs, match_net_type=args.match_net_type,
             absolute_norm=args.absolute_norm)
