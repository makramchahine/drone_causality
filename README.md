# drone_causality

All training, data processing, and analysis code used for the paper "Robust Visual Flight Navigation with Liquid Neural Networks". For code run onboard the drone, see [this repository](https://github.com/GoldenZephyr/rosetta_drone).

## Installation Instructions

For x86 based systems (most computers), setup your python environment using conda environment file in configs/environment.yml

~~~
cd drone_causality
conda env create -f config/environment.yml
conda activate causality
~~~

Another environment file is available for ppc64le (PowerPC) based architectures
~~~
conda env create -f config/satori_environment.yml
conda activate causality
~~~

Alternatively, a Docker image containing all required packages can be found on Docker Hub at dolphonie1/causal_repo:0.1.17

~~~
docker pull dolphonie1/causal_repo:0.1.17
docker run -it --net=host dolphonie1/causal_repo:0.1.17 /bin/bash
~~~
## Downloading Datasets/Existing Checkpoints
The original hand-collected training dataset can be found [here](http://knightridermit.myqnapcloud.com:8080/share.cgi?ssid=06lMJMN&fid=06lMJMN&path=%2F&filename=devens_snowy_fixed.zip&openfolder=forcedownload&ep=) (filename: devens_snowy_fixed, size:33.2GB). Additionally, we have a subset of the full `devens_snowy_fixed` dataset that only contains runs with the chair [here](http://knightridermit.myqnapcloud.com:8080/share.cgi?ssid=06lMJMN&fid=06lMJMN&path=%2F&filename=devens_chair.zip&openfolder=forcedownload&ep=) (devens_chair, 2.3GB).

We have also included the exact synthetic datasets we used for our experiments. These datasets were created using the script at `preprocess/closed_loop_augmentation.py`, but with a random seed. We have both a [full dataset](http://knightridermit.myqnapcloud.com:8080/share.cgi?ssid=06lMJMN&fid=06lMJMN&path=%2F&filename=synthetic_small4.zip&openfolder=forcedownload&ep=) (synthetic_small4, 14.7GB) used to train the starting checkpoint here and a [chair-only dataset](http://knightridermit.myqnapcloud.com:8080/share.cgi?ssid=06lMJMN&fid=06lMJMN&path=%2F&filename=synthetic_chair4.zip&openfolder=forcedownload&ep=) (synthetic_chair, 4.3 GB) used to fine-tune the final models for testing at.

To replicate the results of our experiments, first train on the entire dataset, `devens_snowy_fixed` with the full synthetic dataset `synthetic_small4` or use the checkpoints in chair4_long_balanced.

Afterwards, fine-tune models starting from `checkpoints/chair4_long_balanced` on the `devens_chair` dataset with the synthetic dataset `synthetic_chair4`

All training was done using the best hyperparameters found in the `old_db` folder.
## Training Models
### Training Once
The script tf_data_training.py executes 1 training run. It loads data and models, sets up multi-GPU processing strategy, and runs training while checkpointing models. The script's default hyperparameters are static and are _not_ the best hyperparameters found during parameter tuning. Any hyperparameters need to be manually specified.

Example usage: 
~~~
python3 tf_data_training.py --model ncp --data_dir /path/to/devens_snowy_fixed --extra_data_dir /path/to/synthetic_small4 --epochs 100 --seq_len 64 --data_stride 1 --data_shift 16
~~~

### Training Multiple Times
The convenience script train_multiple.py automatically manages multiple training runs, saving log JSON files to record the results of each run and intelligently determining how many runs have been completed so far to allow for resuming training. The script also automatically loads hyperparameters from the best study when given a hyperparameter study database file.

Example usage:
~~~
python train_multiple.py ncp_objective /path/to/devens_snowy_short --n_trains 5 --batch_size 300 --storage_name sqlite:///old_db/ncp_objective".db --storage_type rdb --timeout 72000 --extra_data_dir /path/to/synthetic --hotstart_dir /path/to/chair4_long_balanced --study_name hyperparam_tuning_ --out_dir chair4_fine_targets
~~~

The `storage_name` argument specifies the database file (in the `old_db` folder) that the best hyperparameters should be read from. Unfortunately, because training was conducted on different machines, different objectives have different hyperparameter files. For each type of network, use the following `storage_name`:

- LSTM: sqlite:///old_db/lstm_objective.db
- CFC: sqlite:///old_db/cfc_objective.db
- NCP: sqlite:///old_db/ncp_objective.db
- GRUODE: sqlite:///old_db/hyperparam_tuning.db
- TCN: old_db/tcn_objective.json
- Wiredcfccell (Sparse-CfC): sqlite:///old_db/wiredcfccell_objective.db
- LTC: sqlite:///old_db/hyperparam_tuning.db
- CT-RNN: old_db/ctrnn_objective.json

Note that the `storage_type` argument should be set to `rdb` for sqlite URLs, json for JSON files, and `pkl` for PKL files

## Preprocessing Data
This section describes the methodology used to generate the dataset `devens_snowy_fixed`.

If using new data collected on the drone, use script `preprocess/process_data.py` to format it correctly for training scripts. Runs should have the red channel as the 0th channel (appear not flipped when opened by an image viewer).

The runs tht don't have an underscore in them (ex 1628106140.64) are the original long runs that see all 5 targets. The runs with underscores (ex 1628106140.64_1) are generated using the script `preprocess/sequence_slice/slice_sequence.py`, which provides a GUI for specifying start and end points and automaticallly copies images and control csv.

To generate new synthetic datasets, use the script `preprocess/closed_loop_augmentation.py`. The directory `preprocess/aug_json` contains JSON files that contain images to be augmented and the pixel location of the target within the image (generated by `preprocess/select_targets.py`).

Example Usage:
The dataset `synthetic_small4` was generated with the following invocation:
~~~
python closed_loop_augmentation.py aug_json/synthetic_full_small.json /path/to/out/dir/synthetic_small4 --num_aug 5 --balance_classes --balance_offsets -10 -70 0 0
~~~

## Tuning Hyperparams
The Optuna hyperparameter study db files in the `old_db` directory were generated using the file `hyperparam_tuning.py`. This script is responsible for sampling parameters using Bayesian Optimization, running training multiple times using the objective functions in `utils/objective_functions.py`, and logging the results within the Optuna study object.

Example usage:
~~~
python hyperparameter_tuning.py ncp_objective /path/to/dataset --n_trials 40 --timeout 64800 --batch_size 300 --extra_data_dir /path/to/synthetic_dataset
~~~

## Analyzing Results

### Stress Tests
The stress test figures used in the paper were generated with the script `analysis/perturb_trajectory.py`

Example usage:
~~~
python analysis/perturb_trajectory.py dataset_jsons/chair_short_raw.json checkpoints/chair4_fine/train/params.json contrast_perturbation --distance_fxn final_distance --deltas 0.5 1.5 2 2.5 --skip_models ctrnn_mixedcfc --perturb_frac 0.2 --force_even_x
~~~

This file, (and most other analysis files), consume a dataset_json file in the format
~~~
{
    "name_of_dataset" : [
        "/path/to/dataset",
        [boolean of whether to flip color channels],
        "path/to/control_csv" or null if no csv desired,
    ], ...
}
~~~

You will most likely have to edit the files in `dataset_jsons` to match the runs you want to analyze on your computer.
### Useful files

- visualization_runner.py: Used for generating videos of visual backprop, input grad, shap, or other visualization technique overlaid on original video sequence and visualization of controls
- analysis/vis_grid.py: Used for generating multiple images of visual backprop and original camera images. Used in paper
- analysis/lipschitz_constant.py: Calculates lipschitz constant of RNN hidden state components when seeing a given sequence of inputs. (Measures maximum difference in rnn hidden state in 2 consecutive timestamps)
- analysis/loss_graph.py: Plots training loss curves
- analysis/ssim.py: Calculates structural similarity index of saliency maps when random noise is added to image


Contact: patrick[dot]d[dot]kao[at]gmail[dot]com for any questions