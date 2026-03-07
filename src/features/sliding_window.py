import numpy as np

def create_sliding_windows(df, sensor_cols, window_size = 30):
    x = []
    y = []

    for engine in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine]

        sensors = engine_df[sensor_cols].values
        rul = engine_df['RUL'].values

        for i in range(len(engine_df)-window_size):
            x.append(sensors[i:i+window_size])
            y.append(rul[i+window_size-1])

    return np.array(x), np.array(y)
