def remove_constant_sensors(df):

    sensor_cols = [col for col in df.columns if "sensor" in col]

    nunique = df[sensor_cols].nunique()

    constant_sensors = nunique[nunique <= 1].index

    df = df.drop(columns=constant_sensors)

    return df

