from sklearn.preprocessing import StandardScaler
def normalize_sensors(df, sensor_cols):
    df_norm = df.copy()
    for cond in df['op_cluster'].unique():
        idx = df['op_cluster'] == cond
        scaler = StandardScaler()
        df_norm.loc[idx, sensor_cols] = scaler.fit_transform(
                df.loc[idx, sensor_cols]
        )
    return df_norm
