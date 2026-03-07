from sklearn.preprocessing import StandardScaler
def normalize_sensors(df, sensor_cols):
    df_norm = df.copy()

    # convert sensors to float
    df_norm[sensor_cols] = df_norm[sensor_cols].astype(float)

    scaler = StandardScaler()

    for cond in df_norm["op_cluster"].unique():
        idx = df_norm["op_cluster"] == cond
        df_norm.loc[idx, sensor_cols] = scaler.fit_transform(
            df_norm.loc[idx, sensor_cols]
        )

    return df_norm
