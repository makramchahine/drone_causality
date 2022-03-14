# drone_causality

Up-to-date fork of deepdrone repo at: https://github.com/mit-drl/deepdrone that focuses on causality experiments for M300

Useful files:
- tf_data_training.py: script that actually executes 1 training run. Loads data and models, sets up multi-GPU processing strategy, and runs training while checkpointing models
- hyperparameter_tuning.py: uses Optuna to automatically run Bayesian optimization for hyperparameter tuning. SQL database allows running multiple optimization sessions in paraallel
- utils/objecitve_functions.py: has all tuning experiments used by hyperparameter_tuning.py, specifying which parameters should be sampled and reasonable ranges to sample from
- train_multiple.py: helper script that runs training n times and saves results in a JSON file using the best params from a hyperparameter tuning experiment
- tf_data_loader.py: converts dataset on disk into keras dataset object
- tf_cfc.py and node_cell.py: external library files (no pip verion exists so source copied)

Contact: patrick[dot]d[dot]kao[at]gmail[dot]com for any questions