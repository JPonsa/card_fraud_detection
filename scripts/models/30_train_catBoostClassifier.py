"""
Create an ML pipeline for the card_fraud dataset, train it and register it.
"""


import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import optuna
from catboost import CatBoostClassifier

import mlflow
from mlflow.data.pandas_dataset import PandasDataset


def get_pipeline(params) -> imbpipeline:
    # Define the pipeline components
    scaler = StandardScaler()
    over = SMOTE(sampling_strategy=0.3)
    under = RandomUnderSampler(sampling_strategy=1)

    model = CatBoostClassifier(**params)

    pipeline = imbpipeline(
        [
            ("preprocessor", scaler),
            ("oversampling", over),
            ("undersampling", under),
            ("classifier", model),
        ]
    )

    return pipeline


def objective(trial):
    # Define the hyperparameters to be tuned
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "random_seed": 42,
        "eval_metric": "Recall",
        "verbose": False,
    }

    # Create the pipeline
    pipeline = get_pipeline(params)

    # Fit the model on the training data
    pipeline.fit(X_train, y_train)

    # Predict on the validation data
    y_pred = pipeline.predict(X_val)

    # Calculate the recall score
    recall = recall_score(y_val, y_pred)

    return recall


def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1}


if __name__ == "__main__":
    # Load preprocessed train and validate data
    preprocessed_path = "./data/preprocessed/"

    train_df = pd.read_csv(preprocessed_path + "train.csv")
    val_df = pd.read_csv(preprocessed_path + "validate.csv")

    dataset_source_url = (
        "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/"
    )
    training: PandasDataset = mlflow.data.from_pandas(
        train_df, source=dataset_source_url
    )

    target = ["TX_FRAUD"]

    y_train = train_df[target].values
    y_val = val_df[target].values

    X_train = train_df.drop(target, axis=1)
    X_val = val_df.drop(target, axis=1)

    # optimise Catboost for max recall

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///card_fraud_catBoostClassifier.optuna.db.sqlite3",
        study_name="card_fraud",
        load_if_exists=True,
    )

    # Optimize the objective function
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    pipeline = get_pipeline(study.best_params)

    mlflow.set_experiment("card_fraud")

    with mlflow.start_run(run_name="catBoostClassifier validation"):
        # Register dataset
        mlflow.log_input(training, context="training")

        pipeline.fit(X_train, y_train)

        # Record Metrics
        y_pred = pipeline.predict(X_val)
        mlflow.log_metrics(get_metrics(y_val, y_pred))

        # Register Model
        mlflow.log_params(pipeline.get_params())
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="models",
            registered_model_name="card_fraud_catBoostClassifier",
        )

mlflow.register_model(name="card_fraud_catBoostClassifier")
