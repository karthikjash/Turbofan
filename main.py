from src.data.data_loader import load_train_data
from src.preprocessing.rul import compute_rul
from src.preprocessing.sensor_cleaning import remove_constant_sensors
from src.preprocessing.op_condition_cluster import cluster_operating_conditions
from src.preprocessing.normalization import normalize_sensors
from src.features.sliding_window import create_sliding_windows
from src.features.pca_reduction import apply_pca

import numpy as np
import os


def main():

    # -----------------------------
    # 1. Load Data
    # -----------------------------
    print("Loading data")
    df = load_train_data("data/raw/train_FD001.txt")

    # -----------------------------
    # 2. Compute RUL
    # -----------------------------
    print("Computing the RUL")
    df = compute_rul(df)

    # -----------------------------
    # 3. Sensor Cleaning
    # -----------------------------
    print("Cleaning sensors")
    df = remove_constant_sensors(df)

    # -----------------------------
    # 4. Operating Condition Clustering
    # -----------------------------
    print("Clustering operational conditions")
    df = cluster_operating_conditions(df)

    # -----------------------------
    # 5. Select Sensor Columns
    # -----------------------------
    sensor_cols = [col for col in df.columns if "sensor" in col]

    print("\nColumns after preprocessing:")
    print(df.columns)

    # -----------------------------
    # 6. Normalize Sensors
    # -----------------------------
    print("Normalizing sensors")
    df = normalize_sensors(df, sensor_cols)

    # -----------------------------
    # 7. Sliding Window Creation
    # -----------------------------
    print("Creating sliding windows")
    X, y = create_sliding_windows(df, sensor_cols)

    print("\nSliding window shapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # -----------------------------
    # 8. Apply PCA
    # -----------------------------
    print("\nApplying PCA")
    X, pca = apply_pca(X, n_components=8)

    print("New shape after PCA:", X.shape)

    # Explained variance (important)
    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio_)

    # -----------------------------
    # 9. Save Processed Data
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv("data/processed/processed_turbofan.csv", index=False)

    np.save("data/processed/X_pca.npy", X)
    np.save("data/processed/y.npy", y)

    print("\nProcessed dataset saved to data/processed/")

    # -----------------------------
    # 10. Debug Output
    # -----------------------------
    print("\nFirst sliding window (X[0]):")
    print(X[0])

    print("\nCorresponding RUL (y[0]):")
    print(y[0])

    print("\nSample dataframe:")
    print(df.head())


if __name__ == "__main__":
    main()

