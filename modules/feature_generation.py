from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .index_slicing import get_cols_idx, get_slice


def get_features_df_and_targets(
    df: pd.DataFrame,
    features_ids,
    targets_ids,
    id_column: Union[str, Sequence[str]] = "store",
    date_column: Union[str, Sequence[str]] = "ds",
    target_column: str = "y",
    oil_column: str = "dcoilwtico"

):
    df = df.copy()
    # Формируем календарные признаки из признаков времени
    df["minute"] = df[date_column].dt.minute
    df["hour"] = df[date_column].dt.hour
    df["day"] = df[date_column].dt.day
    df["month"] = df[date_column].dt.month
    df["quarter"] = df[date_column].dt.quarter
    df["year"] = df[date_column].dt.year
    df['day_of_week'] = df['ds'].dt.dayofweek.astype('int8')
    df['day_of_month'] = df['ds'].dt.day.astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    df['is_salary_day'] = ((df['day_of_month'] == 15) | (df['ds'].dt.is_month_end)).astype('int8')
    date_columns = ["minute", "hour", "day", "month", "quarter", "year", 'day_of_week',
                    'day_of_month', 'is_weekend', 'is_salary_day']

    # Признаки идентификатора ряда и времени возьмем из таргетных индексов
    features_df_id = get_slice(df, (targets_ids, get_cols_idx(df, id_column)))
    features_time = get_slice(df, (targets_ids, get_cols_idx(df, date_columns)))

    # Лаговые признаки возьмем из индексов признаков. Все доступные из истории
    features_lags = get_slice(
        df,
        (features_ids, get_cols_idx(df, target_column)),
    )

    features_oil = get_slice(
        df,
        (features_ids, get_cols_idx(df, oil_column)),
    )

    # Объединим все признаки в один массив
    features = np.hstack([features_df_id, features_time, features_lags, features_oil])
    categorical_features_idx = np.arange(
        features_df_id.shape[1] + features_time.shape[1]
    )  # Отметим категориальные признаки для CatBoost

    # Иначе cb.Pool не работает
    features_obj = features.astype(object)
    for j in categorical_features_idx:
        features_obj[:, j] = features_obj[:, j].astype(str)

    # Сформируем таргеты
    targets = get_slice(df, (targets_ids, get_cols_idx(df, target_column)))
    return features_obj, targets, categorical_features_idx
