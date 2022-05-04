import argparse

import joblib
import optuna
from optuna import Study
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_parallel_coordinate, \
    plot_contour, plot_param_importances, plot_slice


def analyze_study(study_name: str, storage_name: str):
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                direction="minimize")
    print(study.best_trial.params)
    # graph_study(study)


def analyze_local(file_path: str):
    study = joblib.load(file_path)
    print(study.best_trial.params)
    # graph_study(study)


def graph_study(study: Study):
    fig = plot_optimization_history(study)
    fig.show()
    fig2 = plot_intermediate_values(study)
    fig2.show()
    fig3 = plot_parallel_coordinate(study)
    fig3.show()
    fig4 = plot_contour(study)
    fig4.show()
    fig5 = plot_param_importances(study)
    fig5.show()
    fig6 = plot_slice(study)
    fig6.show()
    best_params = study.best_params
    print(best_params)
    print(study.best_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("storage_path", type=str)
    parser.add_argument("--study_name", type=str, default=None)
    args = parser.parse_args()
    if args.study_name is not None:
        analyze_study(args.study_name, args.storage_path)
    else:
        analyze_local(args.storage_path)
