import os
import socket
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")


def configure_mlflow():
    # Use localhost:5000 when MLflow server is running (e.g., in GitHub Actions),
    # otherwise fall back to local ./mlruns directory (e.g., when running in Colab).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        server_running = sock.connect_ex(("127.0.0.1", 5000)) == 0

    if server_running:
        mlflow.set_tracking_uri("http://localhost:5000")
        print("Connected to MLflow tracking server at http://localhost:5000")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        print("Using local MLflow tracking directory: ./mlruns")

    mlflow.set_experiment("wellness-tourism-mlops")


def train_and_evaluate():
    # 1. Load train and test splits produced by prep.py / workflow artifact
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv")["ProdTaken"]
    ytest = pd.read_csv("ytest.csv")["ProdTaken"]
    print(f"Loaded splits -> Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")

    # 2. Define feature groups and preprocessing pipeline
    numeric_features = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome",
    ]
    categorical_features = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation",
    ]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    )

    # Handle class imbalance (~80.7% class 0 vs ~19.3% class 1)
    scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())

    xgb_classifier = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    model_pipeline = make_pipeline(preprocessor, xgb_classifier)

    # 3. Define hyperparameter grid and tune with 5-fold cross-validation
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__colsample_bytree": [0.8, 1.0],
    }

    configure_mlflow()

    with mlflow.start_run(run_name="XGBoost_GridSearch_Training"):
        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)

        # Log every tuned parameter combination as a nested MLflow run
        cv_results = pd.DataFrame(grid_search.cv_results_)
        for idx, row in cv_results.iterrows():
            with mlflow.start_run(run_name=f"candidate_{idx}", nested=True):
                clean_params = {
                    k.replace("xgbclassifier__", ""): v
                    for k, v in row["params"].items()
                }
                mlflow.log_params(clean_params)
                mlflow.log_metric("cv_mean_f1", float(row["mean_test_score"]))
                mlflow.log_metric("cv_std_f1", float(row["std_test_score"]))

        # 4. Evaluate best model on held-out test set
        best_model = grid_search.best_estimator_
        best_params = {
            k.replace("xgbclassifier__", ""): v
            for k, v in grid_search.best_params_.items()
        }
        mlflow.log_params(best_params)

        y_pred = best_model.predict(Xtest)
        y_prob = best_model.predict_proba(Xtest)[:, 1]

        metrics = {
            "test_accuracy": float(accuracy_score(ytest, y_pred)),
            "test_precision": float(precision_score(ytest, y_pred)),
            "test_recall": float(recall_score(ytest, y_pred)),
            "test_f1": float(f1_score(ytest, y_pred)),
            "test_roc_auc": float(roc_auc_score(ytest, y_prob)),
        }
        mlflow.log_metrics(metrics)

        print("\n=== Best Hyperparameters ===")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        print("\n=== Test Set Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification Report:\n", classification_report(ytest, y_pred))

        # 5. Save best model pipeline to deployment directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        print(f"Saved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
