import joblib
import optuna
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_parallel_coordinate, \
    plot_contour, plot_param_importances, plot_slice


def analyze_study(study_name: str, storage_name: str):
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                direction="minimize")
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


def analyze_local(file_path: str):
    study = joblib.load(file_path)
    print(study.best_params)


if __name__ == "__main__":
    # analyze_study("hyperparam_tuning_lstm_objective", "sqlite:////home/dolphonie/Desktop/hyperparam_tuning.db")
    analyze_local("/home/dolphonie/Desktop/studies/mixedcfc.pkl")
    # analyze_study("hyperparam_tuning_mixedcfc_objective", "sqlite:///hyperparam_tuning.db")