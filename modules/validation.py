from typing import Tuple
import numpy as np
import pandas as pd

def expanding_window_validation(
    data: pd.DataFrame,
    model,
    horizon: int,
    history: int,
    start_train_size: int,
    step_size: int,
    id_col: str = "store",
    timestamp_col: str = "ds",
    value_col: str = "y",
) -> pd.DataFrame:
    """
    Валидация с расширяющимся окном обучения.
    """
    res_df_list = []

    unique_timestamps = data[timestamp_col].sort_values().unique()
    n_timestamps = len(unique_timestamps)

    train_start_idx = 0
    train_end_idx = start_train_size

    val_start_idx = train_end_idx - history
    val_end_idx = train_end_idx + horizon

    test_start_idx = val_end_idx - history
    test_end_idx = val_end_idx + horizon

    while test_end_idx <= n_timestamps:
        # Маскируем будущие значения для предотвращения утечки данных
        data_masked = data.copy()
        data_masked[value_col] = data_masked[value_col].where(
            data_masked[timestamp_col] < unique_timestamps[test_start_idx + history],
            np.nan,
        )

        train_mask = data_masked[timestamp_col].between(
            unique_timestamps[train_start_idx], unique_timestamps[train_end_idx - 1]
        )
        val_mask = data_masked[timestamp_col].between(
            unique_timestamps[val_start_idx], unique_timestamps[val_end_idx - 1]
        )
        test_mask = data_masked[timestamp_col].between(
            unique_timestamps[test_start_idx], unique_timestamps[test_end_idx - 1]
        )

        train_data = data_masked[train_mask]
        val_data = data_masked[val_mask]
        test_data = data_masked[test_mask]

        # Обучение и прогноз
        model.fit(train_data, val_data, id_col=id_col, timestamp_col=timestamp_col, value_col=value_col)
        predictions = model.predict(test_data, id_col=id_col, timestamp_col=timestamp_col, value_col=value_col)

        # Восстанавливаем истинные значения и фильтруем только тестовый период (после истории)
        test_data_unmasked = data[test_mask].copy()
        
        cutoff_date = unique_timestamps[val_end_idx]
        test_actuals = test_data_unmasked[test_data_unmasked[timestamp_col] >= cutoff_date].copy()
        
        # --- БЕЗОПАСНАЯ СБОРКА РЕЗУЛЬТАТОВ ЧЕРЕЗ MERGE ---
        # Создаем базовый DF с фактами
        res_df = test_actuals[[id_col, timestamp_col, value_col]].rename(columns={value_col: "true_value"})
        res_df["fold"] = len(res_df_list)

        # Присоединяем прогнозы по ключам ID и Date
        res_df = res_df.merge(
            predictions[[id_col, timestamp_col, "predicted_value"]],
            on=[id_col, timestamp_col],
            how="left"
        )

        # Обработка возможных пропусков (если модель не вернула прогноз на конкретную дату/магазин)
        if res_df["predicted_value"].isna().any():
            res_df["predicted_value"] = res_df["predicted_value"].fillna(0)

        res_df_list.append(res_df)

        # Смещение окна
        train_end_idx += step_size
        val_start_idx = train_end_idx - history
        val_end_idx = train_end_idx + horizon
        test_start_idx = val_end_idx - history
        test_end_idx = val_end_idx + horizon

    return pd.concat(res_df_list).reset_index(drop=True), model