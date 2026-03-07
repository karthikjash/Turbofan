from src.data.data_loader import load_train_data
from src.preprocessing.rul import compute_rul
from src.preprocessing.sensor_cleaning import remove_constant_sensors
from src.preprocessing.op_condition_cluster import cluster_operating_conditions
from src.preprocessing.normalization import normalize_sensors
from src.features.sliding_window import create_sliding_windows
import os
def main():
    print("Loading data")
    df = load_train_data("data/raw/train_FD001.txt")

    print("computing the rul")
    df = compute_rul(df)

    print("cleaning sensors")
    df = remove_constant_sensors(df)

    print("Clustering operational conditions")
    df = cluster_operating_conditions(df)

    sensor_cols = [col for col in df.columns if "sensor" in col]
    print(df.columns)

    print("Normalizing sensors")
    df = normalize_sensors(df, sensor_cols)

    print("creating sliding windows")
    x,y = create_sliding_windows(df, sensor_cols)
    

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed_turbofan.csv", index=False)

    print("Processed dataset saved to data/processed/processed_turbofan.csv")

    print("\nFirst sliding window (X[0]):")
    print(x[0])

    print("\nCorresponding RUL (y[0]):")
    print(y[0])

    print(df.head)

if __name__ == "__main__":
    main()
