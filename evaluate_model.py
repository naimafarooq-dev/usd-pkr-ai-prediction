from data_manager import load_data
from model import create_model, row_to_features
from features import create_features
from metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_hit_rate,
)


def evaluate_model(df):
    # Create features
    feature_data = create_features(df)

    # Remove incomplete rows
    feature_data = feature_data.dropna(
        subset=[
            "lag_1",
            "lag_2",
            "rate_change",
            "rate_change_pct",
            "rolling_mean_5",
            "rolling_std_5",
            "hour",
            "minute",
        ]
    ).reset_index(drop=True)

    # Use first 80% for training
    split_index = int(len(feature_data) * 0.80)

    train_data = feature_data.iloc[:split_index]
    test_data = feature_data.iloc[split_index:]

    # Create a fresh ML model
    model = create_model()

    # Train only on the past
    for _, row in train_data.iterrows():

        x = row_to_features(row)
        y = float(row["exchange_rate"])

        model.learn_one(x, y)

    # Store actual and predicted values
    actuals = []
    ml_predictions = []
    baseline_predictions = []

    # Walk forward through test data
    for _, row in test_data.iterrows():

        x = row_to_features(row)

        # ML prediction
        ml_prediction = model.predict_one(x)

        # Simple baseline:
        # predict that the next rate will be the previous rate
        baseline_prediction = float(row["lag_1"])

        # Actual value
        actual = float(row["exchange_rate"])

        ml_predictions.append(ml_prediction)
        baseline_predictions.append(baseline_prediction)
        actuals.append(actual)

        # Learn AFTER prediction
        model.learn_one(x, actual)

    # ML metrics
    ml_mae = calculate_mae(
        actuals,
        ml_predictions
    )

    ml_rmse = calculate_rmse(
        actuals,
        ml_predictions
    )

    ml_hit_rate = calculate_hit_rate(
        actuals,
        ml_predictions
    )

    # Baseline metrics
    baseline_mae = calculate_mae(
        actuals,
        baseline_predictions
    )

    baseline_rmse = calculate_rmse(
        actuals,
        baseline_predictions
    )

    baseline_hit_rate = calculate_hit_rate(
        actuals,
        baseline_predictions
    )

    return (
        ml_mae,
        ml_rmse,
        ml_hit_rate,
        baseline_mae,
        baseline_rmse,
        baseline_hit_rate,
        len(train_data),
        len(test_data),
    )


if __name__ == "__main__":

    df = load_data()

    if len(df) < 20:

        print("Not enough data for evaluation.")

    else:

        (
            ml_mae,
            ml_rmse,
            ml_hit_rate,
            baseline_mae,
            baseline_rmse,
            baseline_hit_rate,
            train_rows,
            test_rows,
        ) = evaluate_model(df)

        print()
        print("========== MODEL EVALUATION ==========")

        print("Training rows:", train_rows)
        print("Testing rows:", test_rows)

        print()
        print("----- ML MODEL -----")
        print(f"MAE: {ml_mae:.6f} PKR")
        print(f"RMSE: {ml_rmse:.6f} PKR")
        print(
            f"Hit Rate (±0.15 PKR): {ml_hit_rate:.2f}%"
        )

        print()
        print("----- NAIVE BASELINE -----")
        print(f"MAE: {baseline_mae:.6f} PKR")
        print(f"RMSE: {baseline_rmse:.6f} PKR")
        print(
            f"Hit Rate (±0.15 PKR): {baseline_hit_rate:.2f}%"
        )

        print()
        print("======================================")