import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Chosen parameters after testing reasonable values from the assignment range.
WINDOW_SIZE = 500
Q_PERCENTILE = 91.0


def sliding_window_predict(x, window_size=WINDOW_SIZE, q=Q_PERCENTILE):
    """
    Upper-tail anomaly detector using fixed-size overlapping windows (step size = 1).

    First window:
        - Compute one threshold from x[0:window_size]
        - Label every point in that first window.

    Every later window:
        - Recompute the percentile threshold using ONLY values in the current window.
        - Label ONLY the newly added point (the final point in that window).

    A point is an anomaly when value >= current q-percentile threshold.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < window_size:
        raise ValueError("Window size cannot be larger than the number of observations.")

    predictions = np.zeros(n, dtype=int)
    thresholds = np.full(n, np.nan, dtype=float)

    # Window 1 covers [0, W-1].
    t1 = np.percentile(x[0:window_size], q, method="linear")
    predictions[0:window_size] = (x[0:window_size] >= t1).astype(int)
    thresholds[0:window_size] = t1

    # Each later window shifts by exactly one observation.
    # For end = W, the window is [1, W] and the new point is index W.
    for end in range(window_size, n):
        start = end - window_size + 1
        current_window = x[start:end + 1]

        threshold = np.percentile(current_window, q, method="linear")
        thresholds[end] = threshold

        # Label only the newly added point.
        predictions[end] = int(x[end] >= threshold)

    return predictions, thresholds


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    return tp, fp, fn, tn


def main():
    parser = argparse.ArgumentParser(
        description="HW2: Overlapping Sliding Windows Anomaly Detection"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="AG_NO3_fill_cells_remove_NAN.csv",
        help="Path to the nitrate CSV file.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    # The provided CSV stores nitrate in NO3N and ground truth in Student_Flag.
    nitrate_col = "NO3N"
    truth_col = "Student_Flag"

    if nitrate_col not in df.columns:
        raise KeyError(
            f"Expected nitrate column '{nitrate_col}'. Found: {list(df.columns)}"
        )
    if truth_col not in df.columns:
        raise KeyError(
            f"Expected ground-truth column '{truth_col}'. Found: {list(df.columns)}"
        )

    # The supplied file is already cleaned. This check prevents silent NaN handling.
    if df[nitrate_col].isna().any():
        raise ValueError(
            "NaNs were found in the nitrate column. "
            "This script expects the provided cleaned dataset."
        )

    x = df[nitrate_col].to_numpy(dtype=float)
    y_true = df[truth_col].to_numpy(dtype=int)

    y_pred, thresholds = sliding_window_predict(
        x,
        window_size=WINDOW_SIZE,
        q=Q_PERCENTILE,
    )

    tp, fp, fn, tn = confusion_counts(y_true, y_pred)

    total_anomalies = tp + fn
    total_normals = tn + fp

    normal_accuracy = tn / total_normals
    anomaly_accuracy = tp / total_anomalies

    print(f"Rows: {len(df)}")
    print(f"Ground-truth anomalies: {total_anomalies}")
    print(f"Window size W: {WINDOW_SIZE}")
    print(f"Percentile q: {Q_PERCENTILE}")
    print("Rule: upper-tail anomaly when value >= current threshold")
    print()
    print(f"TP = {tp}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TN = {tn}")
    print(f"Normal event detection accuracy = {normal_accuracy:.4f} ({normal_accuracy:.2%})")
    print(f"Anomaly event detection accuracy = {anomaly_accuracy:.4f} ({anomaly_accuracy:.2%})")

    # Save predictions so results can be inspected.
    results = df.copy()
    results["Threshold"] = thresholds
    results["Prediction"] = y_pred
    results.to_csv("hw2_predictions.csv", index=False)

    # Plot the entire series and mark detected anomalies.
    if "Date" in df.columns:
        plot_x = pd.to_datetime(df["Date"], errors="coerce")
        x_label = "Date"
    else:
        plot_x = np.arange(len(df))
        x_label = "Index"

    detected = y_pred == 1

    plt.figure(figsize=(14, 6))
    plt.plot(plot_x, x, linewidth=0.8, label="Nitrate (NO3N)")
    plt.scatter(
        np.asarray(plot_x)[detected],
        x[detected],
        s=12,
        marker="o",
        label="Detected anomaly",
    )
    plt.xlabel(x_label)
    plt.ylabel("NO3N")
    plt.title(
        f"Sliding-Window Nitrate Anomaly Detection "
        f"(W={WINDOW_SIZE}, q={Q_PERCENTILE})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("hw2_anomaly_plot.png", dpi=200)
    plt.close()

    print()
    print("Saved: hw2_predictions.csv")
    print("Saved: hw2_anomaly_plot.png")


if __name__ == "__main__":
    main()
