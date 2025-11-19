from datetime import datetime, timedelta

def get_top_n_products(df, days, n):
    cutoff = datetime.now() - timedelta(days=days)
    window = df[df["timestamp"] >= cutoff]

    grouped = (
        window.groupby("product_id")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )

    grouped = grouped[grouped["rating_count"] >= 5]  # optional quality filter

    return grouped.sort_values("avg_rating", ascending=False).head(n)
