import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"


def prepare_data():
    # 1. Load dataset directly from the repository data folder
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Remove unnecessary identifier/index columns
    unnecessary_cols = [col for col in ["Unnamed: 0", "CustomerID"] if col in df.columns]
    df = df.drop(columns=unnecessary_cols)
    print(f"Dropped unnecessary columns: {unnecessary_cols}")

    # 3. Clean data anomalies and handle any missing values
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df = df.dropna().reset_index(drop=True)

    # 4. Separate features (X) and target (y)
    target_col = "ProdTaken"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Perform stratified train-test split (80% train, 20% test)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Save splits locally in the root directory for GitHub Actions artifact upload
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("=== Data Preparation Complete ===")
    print(f"Xtrain shape: {Xtrain.shape}, ytrain shape: {ytrain.shape}")
    print(f"Xtest shape : {Xtest.shape}, ytest shape : {ytest.shape}")
    print("Saved files : Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    prepare_data()
