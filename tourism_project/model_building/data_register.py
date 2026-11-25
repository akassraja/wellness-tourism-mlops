# data_register.py: Responsible for validating and registering the input dataset
# inside the repository before starting the processing/training pipeline.

import os
import sys
import pandas as pd

# Define the local relative path to the source tourism dataset
DATA_PATH = "tourism_project/data/tourism.csv"

# Expected schema representing all necessary attributes for predictive modeling
EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def register_dataset():
    """
    Validates the existence, completeness, and schema integrity of the 
    tourism.csv dataset, and outputs details to verify compliance.
    """
    # 1. Check if dataset file exists in the designated directory
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    # 2. Load the CSV file into a pandas DataFrame
    df = pd.read_csv(DATA_PATH)

    # 3. Verify that the file is not empty
    if df.empty:
        print("ERROR: Dataset is empty.")
        sys.exit(1)

    # 4. Map columns to detect missing and extra headers
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    # 5. Halt the process if essential fields are absent
    if missing_cols:
        print(f"ERROR: Missing expected columns: {missing_cols}")
        sys.exit(1)

    # 6. Print validation diagnostics and statistics
    print("=== Dataset Registration & Validation Successful ===")
    print(f"File Path       : {DATA_PATH}")
    print(f"Shape           : {df.shape[0]} rows, {df.shape[1]} columns")
    if extra_cols:
        print(f"Extra Columns   : {extra_cols}")
    
    # 7. Provide summary of data types and empty cells for initial inspection
    print("\n--- Column Data Types & Missing Values ---")
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isnull().sum()
    })
    print(summary)
    
    # 8. Output class distribution metrics for the target label
    print("\n--- Target Variable Distribution (ProdTaken) ---")
    print(df["ProdTaken"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")


if __name__ == "__main__":
    register_dataset()
