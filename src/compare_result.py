import pandas as pd


RESULT_FILES = [
    "testing/rnn_results.csv",
    "testing/gru_results.csv",
    "testing/lstm_results.csv"
]


# Combine all model result files
results_df = pd.concat(
    [pd.read_csv(file) for file in RESULT_FILES],
    ignore_index=True
)

results_df.to_csv("testing/model_results.csv", index=False)


# Create average comparison table
comparison_df = (
    results_df
    .groupby("model")
    .agg(
        avg_mae=("mae", "mean"),
        avg_rmse=("rmse", "mean"),
        avg_r2=("r2", "mean"),
        avg_baseline_rmse=("baseline_rmse", "mean")
    )
    .reset_index()
    .sort_values("avg_rmse")
)

comparison_df["beats_baseline"] = (
    comparison_df["avg_rmse"] < comparison_df["avg_baseline_rmse"]
)

comparison_df.to_csv("testing/model_comparison.csv", index=False)


print("\nCombined Model Results")
print(results_df)

print("\nModel Comparison")
print(comparison_df)

print("\nSaved:")
print("testing/model_results.csv")
print("testing/model_comparison.csv")