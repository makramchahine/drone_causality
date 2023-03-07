import argparse
import json
import random
from collections import defaultdict

from joblib import Parallel, delayed
from tqdm import tqdm

from mixed_aug import *
from preprocess.aug_utils import save_processsed_seq
from synthetic_aug import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_class_map(all_params):
    class_map = defaultdict(list)
    for params, aug_class in all_params:
        class_map[aug_class].append(params)
    return class_map


def augment_image_list(img_data_path: str, out_path: str, num_aug: int, balance_classes: bool = True,
                       balance_offsets: Optional[Sequence[int]] = None, parallel_execution: bool = True,
                       dry_run: bool = False, *args, **kwargs):
    """
    Augments all images and target locations found in a json at img_data_path. For param meanings, see generate_sequence
    """
    # TODO: get this function to work with mixed aug
    with open(os.path.join(SCRIPT_DIR, img_data_path), "r") as f:
        img_data = json.load(f)

    print("Generating augmentation params")

    def perform_single_aug(i, params):
        seq_out = os.path.join(out_path, f"{i}")
        if os.path.exists(seq_out):
            print(f"Found out path {seq_out}, skipping")
            return

        out_seq, control_inputs = generate_synthetic_sequence(**params)

        save_processsed_seq(seq_out, out_seq, control_inputs, process_seq=True)

    # get aug params
    all_params = []
    for img_path, target_loc in img_data:
        for _ in range(num_aug):
            params = get_synthetic_params(img_path, target_loc, **kwargs)
            if params is not None:
                all_params.append(params)
    print(f"Success rate: {len(all_params) / num_aug / len(img_data)}")

    if balance_classes:
        class_map = get_class_map(all_params)
        print(f"Num before balancing: {len(all_params)}")
        all_params = []
        min_instances = min([len(params_list) for class_name, params_list in class_map.items()])
        for aug_class, params_list in class_map.items():
            offset = int(balance_offsets[aug_class]) if balance_offsets is not None else 0
            selected_params = random.sample(params_list, min_instances + offset)
            all_params.extend([(params, aug_class) for params in selected_params])

    # print aug stats
    print("Class stats")
    new_class_map = get_class_map(all_params)
    for aug_class, params_list in sorted(new_class_map.items()):
        print(f"Instances of {aug_class}: {len(params_list)}")

    mean_x_offset = 0
    mean_y_offset = 0
    for params, aug_class in all_params:
        x_loc, y_loc = params["start_offset"]
        mean_x_offset += x_loc
        mean_y_offset += y_loc

    mean_x_offset /= len(all_params)
    mean_y_offset /= len(all_params)
    print(f"Mean x offset: {mean_x_offset} pixels")
    print(f"Mean y offset: {mean_y_offset} pixels")

    print(f"Number source images: {len(img_data)}")
    print(f"Number augmented images: {len(all_params)}")
    print(f"Aug to source ratio: {len(all_params) / len(img_data):.2f}")

    # perform augmentation in parallel
    if not dry_run:
        print("Starting generation of synthetic images")
        if parallel_execution:
            Parallel(n_jobs=6)(
                delayed(perform_single_aug)(i, params) for i, (params, aug_class) in
                enumerate(tqdm(all_params)))
        else:
            for i, (params, aug_class) in enumerate(tqdm(all_params)):
                perform_single_aug(i, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_data_path", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--num_aug", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--balance_classes", action="store_true")
    parser.add_argument('--balance_offsets', nargs='+', default=None)

    args = parser.parse_args()

    # full images
    # augment_image_list(aug_fn=augment_image_synthetic, img_data_path=args.img_data_path, out_path=args.out_path,
    #                    seq_len=args.seq_len, num_aug=args.num_aug, min_x_offset=90, max_x_offset=180,
    #                    min_y_offset=60,
    #                    max_y_offset=120, )
    augment_image_list(img_data_path=args.img_data_path, out_path=args.out_path,
                       num_aug=args.num_aug, balance_classes=args.balance_classes, balance_offsets=args.balance_offsets,
                       parallel_execution=True, dry_run=args.dry_run, min_x_offset=10, max_x_offset=70,
                       min_y_offset=5, max_y_offset=40, max_zoom=2,
                       frame_size_padding=50, min_static_fraction=0.2, max_static_fraction=0.35, min_seq_len=120,
                       max_seq_len=250, turn_channel=TurnChannel.YAW, lateral_motion=True)
    # augment_image_list(aug_fn=augment_image_mixed, img_data_path=args.img_data_path, out_path=args.out_path,
    #                    num_aug=args.num_aug, min_x_offset=30, max_x_offset=60, min_y_offset=20,
    #                    max_y_offset=40, )
