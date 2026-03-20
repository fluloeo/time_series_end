from pathlib import Path

SEED = 42

# отображение
PANDAS_FLOAT_FORMAT = '{:.2f}'.format

# предупреждения
IGNORE_WARNINGS = True


# пути
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")

# создаём папки (можно оставить тут — это нормально)
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# параметры модели
H = 16
NWRMSLE_WEIGHT = 1.25

# типы данных
DTYPES = {
    'store_nbr': 'int8',
    'item_nbr': 'int32',
    'onpromotion': 'bool',
    'unit_sales': 'float32'
}