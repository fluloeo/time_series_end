from src.utils.setup import setup_environment, seed_everything
from config import SEED

# data
from src.data.load_data import load_train, load_dense

# features
from src.features.build_dataset import build_dense_dataset, save_dense_dataset
from src.features.build_features import (
    load_base_dataset,
    load_external_data,
    build_variant_a,
    build_variant_b,
    build_variant_c,
    save_variants
)

# models
from src.models.baselines import run_baselines
from src.models.train_catboost import run_catboost


def main():
    # ===================== SETUP =====================
    setup_environment()
    seed_everything()

    # 1. LOAD RAW DATA 
    print("\n=== LOAD RAW DATA ===")
    train = load_train()

    # 2. BUILD DENSE DATASET 
    print("\n=== BUILD DENSE DATASET ===")
    dense = build_dense_dataset(train)

    print(f"Dense shape: {dense.shape}")
    print(f"Unique series: {dense['unique_id'].nunique()}")

    path = save_dense_dataset(dense, 'train_dense_3stores_100items.csv')
    print(f"Saved to: {path}")

    # 3. BASELINES 
    print("\n=== BASELINES ===")
    df_dense = load_dense()
    pivot_baseline, _ = run_baselines(df_dense)

    print("\nBaseline results:")
    print(pivot_baseline)

    # 4. FEATURE ENGINEERING 
    print("\n=== FEATURE ENGINEERING ===")

    base = load_base_dataset()
    stores, items, oil, holidays, transactions = load_external_data()

    df_a = build_variant_a(base, stores, items)
    df_b = build_variant_b(df_a, transactions)
    df_c = build_variant_c(df_b, oil, holidays)

    save_variants(df_a, df_b, df_c)

    # 5. CATBOOST 
    print("\n=== CATBOOST TRAINING ===")
    pivot_cb, _ = run_catboost()

    print("\nCatBoost results:")
    print(pivot_cb)


if __name__ == "__main__":
    main()