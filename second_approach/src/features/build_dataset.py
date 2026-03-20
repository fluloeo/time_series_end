import pandas as pd
from config import DATA_DIR


def build_dense_dataset(train, n_stores=3, n_items=100):
    # ===================== ВЫБОР ТОПОВ =====================
    top_stores = (
        train.groupby('store_nbr')['unit_sales']
        .sum()
        .nlargest(n_stores)
        .index.tolist()
    )

    top_items = (
        train.groupby('item_nbr')['unit_sales']
        .sum()
        .nlargest(n_items)
        .index.tolist()
    )

    # Фильтрация
    sample = train[
        (train['store_nbr'].isin(top_stores)) &
        (train['item_nbr'].isin(top_items))
    ].copy()

    # ===================== ПЛОТНЫЙ РЯД =====================
    all_dates = pd.date_range(
        start=train['date'].min(),
        end=train['date'].max(),
        freq='D'
    )

    stores_items = sample[['store_nbr', 'item_nbr']].drop_duplicates()

    grid = (stores_items.assign(key=1)
            .merge(pd.DataFrame({'date': all_dates, 'key': 1}), on='key')
            .drop(columns='key'))

    dense = grid.merge(
        sample[['store_nbr', 'item_nbr', 'date', 'unit_sales', 'onpromotion']],
        on=['store_nbr', 'item_nbr', 'date'],
        how='left'
    )

    dense['unit_sales'] = dense['unit_sales'].fillna(0).clip(lower=0)
    dense['onpromotion'] = dense['onpromotion'].fillna(False)

    # ===================== УНИКАЛЬНЫЙ ИНДЕКС =====================
    dense['unique_id'] = dense['store_nbr'].astype(str) + '_' + dense['item_nbr'].astype(str)
    dense = dense.rename(columns={'date': 'ds', 'unit_sales': 'y'})

    dense = dense.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    return dense


def save_dense_dataset(dense, filename='train_dense.csv'):
    output_file = DATA_DIR / filename
    dense[['unique_id', 'ds', 'y', 'onpromotion']].to_csv(output_file, index=False)
    return output_file