assert dense.groupby('unique_id').size().nunique() == 1, "Ряды имеют разную длину"

assert dense['y'].isna().sum() == 0, "Есть пропуски в target"

def validate_dense(dense):
    return {
        "n_series": dense['unique_id'].nunique(),
        "lengths": dense.groupby('unique_id').size().unique(),
        "y_min": dense['y'].min(),
        "y_max": dense['y'].max(),
        "y_na": dense['y'].isna().sum(),
        "promo_na": dense['onpromotion'].isna().sum(),
    }