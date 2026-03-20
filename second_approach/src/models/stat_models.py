import pandas as pd
import numpy as np
from time import time

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta

from config import DATA_DIR, RESULTS_DIR, H
from src.models.metrics import nwrmsle


def run_stat_models(filename='train_dense_3stores_100items.csv', min_history=30):
    # ===================== ЗАГРУЗКА =====================
    df = pd.read_csv(DATA_DIR / filename, parse_dates=['ds'])

    # ===================== ФИЛЬТР КОРОТКИХ РЯДОВ =====================
    lengths = df.groupby('unique_id').size()
    valid_uids = lengths[lengths >= min_history].index
    df_filtered = df[df['unique_id'].isin(valid_uids)]

    print(
        f"После фильтра ≥ {min_history} точек: "
        f"{df_filtered['unique_id'].nunique()} рядов из {df['unique_id'].nunique()}"
    )

    # ===================== МОДЕЛИ =====================
    models = [
        AutoETS(),
        AutoTheta()
    ]

    # ===================== EXPANDING WINDOW =====================
    end_dates = pd.date_range(start='2017-06-15', end='2017-07-31', freq='14D')

    windows = []
    for end in end_dates:
        train_end = end
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=H - 1)

        windows.append({
            'name': f"expand_{end.date()}",
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end
        })

    # ===================== ОБУЧЕНИЕ =====================
    results = []

    for w in windows:
        print(f"\n→ {w['name']}")

        train = df_filtered[df_filtered['ds'] <= w['train_end']]
        val = df_filtered[
            (df_filtered['ds'] >= w['val_start']) &
            (df_filtered['ds'] <= w['val_end'])
        ]

        # фикс бага
        if w['val_end'] > df_filtered['ds'].max():
            continue

        if len(train) == 0 or len(val) == 0:
            print("   → пустое окно")
            continue

        start_time = time()

        sf = StatsForecast(models=models, freq='D', n_jobs=-1)
        sf.fit(train)

        print(f"   fit занял {time() - start_time:.1f} сек")

        forecast = sf.predict(h=H)

        val_pred = forecast.reset_index().merge(
            val[['unique_id', 'ds', 'y']],
            on=['unique_id', 'ds'],
            how='left'
        )

        if val_pred['y'].isna().any():
            print("  есть NaN в y_true")

        # ===================== МЕТРИКА =====================
        for model_name in ['AutoETS', 'AutoTheta']:
            y_true = val_pred['y'].values
            y_pred = val_pred[model_name].values

            # защита от отрицательных прогнозов
            y_pred = np.clip(y_pred, 0, None)

            score = nwrmsle(y_true, y_pred)

            results.append({
                'window': w['name'],
                'model': model_name,
                'nwrmsle': round(score, 5),
                'train_days': len(train['ds'].unique())
            })

    # ===================== РЕЗУЛЬТАТЫ =====================
    res_df = pd.DataFrame(results)

    pivot = res_df.pivot(index='model', columns='window', values='nwrmsle').round(5)
    pivot['mean'] = pivot.mean(axis=1).round(5)

    print("\n=== Auto-модели, Expanding Window, NWRMSLE ===")
    print(pivot)

    # ===================== СОХРАНЕНИЕ =====================
    RESULTS_DIR.mkdir(exist_ok=True)

    pivot.to_csv(RESULTS_DIR / "stat_models_results.csv")
    res_df.to_csv(RESULTS_DIR / "stat_models_results_raw.csv", index=False)

    print(f"\nРезультаты сохранены в папку: {RESULTS_DIR}")

    return pivot, res_df