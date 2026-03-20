import pandas as pd
from config import DATA_DIR, DTYPES

def load_train():
    train = pd.read_csv(
        DATA_DIR / "train_2017.csv",
        dtype=DTYPES,
        parse_dates=['date']
    )
    return train

def load_dense(filename='train_dense_3stores_100items.csv'):
    import pandas as pd
    from config import DATA_DIR
    
    return pd.read_csv(DATA_DIR / filename, parse_dates=['ds'])