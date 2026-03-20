import pandas as pd
from config import DATA_DIR, RESULTS_DIR


# ===================== ЗАГРУЗКА =====================
def load_base_dataset(filename='train_dense_3stores_100items.csv'):
    df = pd.read_csv(DATA_DIR / filename, parse_dates=['ds'])

    # восстановление store/item
    df['store_nbr'] = df['unique_id'].str.split('_').str[0].astype('int8')
    df['item_nbr'] = df['unique_id'].str.split('_').str[1].astype('int32')

    return df


def load_external_data():
    stores = pd.read_csv(DATA_DIR / 'stores.csv')
    items = pd.read_csv(DATA_DIR / 'items.csv')
    oil = pd.read_csv(DATA_DIR / 'oil.csv', parse_dates=['date'])
    holidays = pd.read_csv(DATA_DIR / 'holidays_events.csv', parse_dates=['date'])
    transactions = pd.read_csv(DATA_DIR / 'transactions.csv', parse_dates=['date'])

    return stores, items, oil, holidays, transactions


# ===================== VARIANT A =====================
def build_variant_a(base, stores, items):
    df = base.copy()

    # Date features
    df['dow'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['month'] = df['ds'].dt.month
    df['is_payday'] = (
        (df['ds'].dt.day == 15) |
        (df['ds'].dt.day == df['ds'].dt.days_in_month)
    ).astype(int)

    # Meta
    df = df.merge(
        stores[['store_nbr', 'cluster', 'city', 'state', 'type']],
        on='store_nbr', how='left'
    )

    df = df.merge(
        items[['item_nbr', 'family', 'class', 'perishable']],
        on='item_nbr', how='left'
    )

    return df


# ===================== VARIANT B =====================
def build_variant_b(df_a, transactions):
    df = df_a.copy()

    transactions = transactions.rename(columns={'date': 'ds'})

    df = df.merge(
        transactions[['ds', 'store_nbr', 'transactions']].shift(1),
        on=['ds', 'store_nbr'],
        how='left'
    )

    df['transactions'] = df['transactions'].fillna(0)

    group = df.groupby('unique_id')

    # лаги
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = group['y'].shift(lag)

    # rolling
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = group['y'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = group['y'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    df['trans_rolling_mean_7'] = group['transactions'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )

    df = df.dropna()

    return df


# ===================== VARIANT C =====================
def build_variant_c(df_b, oil, holidays):
    df = df_b.copy()

    # Oil
    oil = oil.rename(columns={'date': 'ds'})
    df = df.merge(oil, on='ds', how='left')

    for lag in [1, 7]:
        df[f'oil_lag_{lag}'] = df.groupby('unique_id')['dcoilwtico'].shift(lag)

    df['oil_rolling_mean_7'] = df['dcoilwtico'].shift(1).rolling(7, min_periods=1).mean()

    # Holidays
    holidays = holidays.rename(columns={'date': 'ds'})

    holidays_simple = holidays[['ds']].copy()
    holidays_simple['is_holiday'] = 1
    holidays_simple = holidays_simple.drop_duplicates(subset=['ds'])

    holiday_locale = holidays.groupby('ds')['locale'].first().reset_index()

    holidays_simple = holidays_simple.merge(holiday_locale, on='ds', how='left')
    holidays_simple = holidays_simple.rename(columns={'locale': 'holiday_locale'})

    holidays_simple['holiday_locale'] = holidays_simple['holiday_locale'].fillna('NoHoliday')

    df = df.merge(holidays_simple, on='ds', how='left')

    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    df['holiday_locale'] = df['holiday_locale'].fillna('NoHoliday').astype(str)

    df = df.dropna(subset=['y', 'onpromotion'])

    return df


# ===================== СОХРАНЕНИЕ =====================
def save_variants(df_a, df_b, df_c):
    RESULTS_DIR.mkdir(exist_ok=True)

    path_a = RESULTS_DIR / 'catboost_variantA_base.csv'
    path_b = RESULTS_DIR / 'catboost_variantB_time.csv'
    path_c = RESULTS_DIR / 'catboost_variantC_full.csv'

    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    df_c.to_csv(path_c, index=False)

    print("\nГотово! Файлы:")
    print(f"• A → {path_a}")
    print(f"• B → {path_b}")
    print(f"• C → {path_c}")