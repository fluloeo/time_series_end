import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive

from config import H, RESULTS_DIR
from src.models.metrics import nwrmsle


def run_baselines(df):
    """
    Запуск baseline моделей (Naive, SeasonalNaive)
    с expanding window валидацией.
    """

    print(f"Загружено рядов: {df['unique_id'].nunique()}, строк: {len(df):,}")

    # ===================== РАСШИРЯЮЩЕЕСЯ ОКНО =====================
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

    print(f"Создано {len(windows)} expanding окон")

    # ===================== МОДЕЛИ =====================
    models = [Naive(), SeasonalNaive(season_length=7)]

    # ===================== ОБУЧЕНИЕ =====================
    results = []

    for w in windows:
        print(f"\n→ Окно: {w['name']}")

        train = df[df['ds'] <= w['train_end']]
        val = df[(df['ds'] >= w['val_start']) & (df['ds'] <= w['val_end'])]

        # защита от выхода за границы данных
        if w['val_end'] > df['ds'].max():
            continue

        sf = StatsForecast(models=models, freq='D', n_jobs=-1)
        sf.fit(train)
        forecast = sf.predict(h=H)

        val_pred = forecast.reset_index().merge(
            val[['unique_id', 'ds', 'y']],
            on=['unique_id', 'ds'],
            how='left'
        )

        if val_pred['y'].isna().any():
            print("  есть NaN в y_true")

        # ===================== МЕТРИКА =====================
        for model_name in ['Naive', 'SeasonalNaive']:
            y_true = val_pred['y'].values
            y_pred = val_pred[model_name].values

            score = nwrmsle(y_true, y_pred)

            results.append({
                'window': w['name'],
                'model': model_name,
                'nwrmsle': round(score, 5),
                'train_days': len(train['ds'].unique())
            })

    # ===================== ИТОГИ =====================
    res_df = pd.DataFrame(results)

    pivot = res_df.pivot(index='model', columns='window', values='nwrmsle').round(5)
    pivot['mean'] = pivot.mean(axis=1).round(5)

    print("\n=== РЕЗУЛЬТАТЫ EXPANDING WINDOW (NWRMSLE) ===")
    print(pivot)

    # ===================== СОХРАНЕНИЕ =====================
    RESULTS_DIR.mkdir(exist_ok=True)

    pivot.to_csv(RESULTS_DIR / "baseline_results.csv")
    res_df.to_csv(RESULTS_DIR / "baseline_results_raw.csv", index=False)

    print(f"\nРезультаты сохранены в папку: {RESULTS_DIR}")

    return pivot, res_df