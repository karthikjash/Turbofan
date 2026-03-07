import pandas as pd
def load_train_data(path):
    cols = ['engine_id','cycle','op1','op2','op3'] + \
           [f'sensor{i}' for i in range(1,22)]

    df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None
            )
    df = df.iloc[:, :26]
    df.columns = cols
    return df

