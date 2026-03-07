def remove_constant_sensors(df):
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index
    df = df.drop(columns=constant_cols)
    return df
