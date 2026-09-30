# HW2 — Overlapping Sliding Windows Anomaly Detection

## Method

I used a fixed-size, overlapping sliding-window threshold detector on the `NO3N` nitrate column. The window advances by exactly one sample each time.

My final parameters were:

- **Window size:** `W = 500`
- **Percentile:** `q = 91.0`
- **Rule:** upper-tail, one-sided detection
- A point is labeled an anomaly when `value >= current q-percentile threshold`.

For the first window, I compute

```python
T1 = np.percentile(x[0:W], q, method="linear")
```

and label **all points in the first window** using `T1`.

For every later window, I recompute the threshold using only the values inside that current window. I then label **only the newly added point**, which is the last point in that window. The step size is always 1.

The ground-truth `Student_Flag` column is used only to evaluate the predictions; it is not used to compute any threshold.

## Why I chose W = 500 and q = 91

I tested several window sizes and percentile values in the range suggested by the assignment. A window of 500 samples was responsive enough to adapt to local changes in the nitrate time series. Larger windows produced smoother thresholds, but they missed too many of the labeled anomaly events.

For `W = 500`, increasing `q` reduced false positives but also reduced anomaly recall. I selected `q = 91.0` because it was the highest simple percentile I tested that still met both required performance targets. For comparison:

| W | q | Normal accuracy | Anomaly accuracy |
|---:|---:|---:|---:|
| 500 | 90 | 84.73% | 75.89% |
| **500** | **91** | **85.76%** | **75.18%** |
| 500 | 92 | 87.00% | 73.05% |
| 1000 | 90 | 86.86% | 71.63% |
| 1500 | 90 | 85.66% | 71.63% |
| 2000 | 90 | 85.94% | 72.34% |

`W = 500, q = 91` gives a useful tradeoff: it improves normal-event accuracy over `q = 90` while still keeping anomaly-event accuracy above the required 75%.

## Results

The dataset contains **30,790 total observations**, including exactly **141 ground-truth anomalies**.

Using `W = 500` and `q = 91.0`:

- **TP = 106**
- **FP = 4363**
- **FN = 35**
- **TN = 26286**

The two required accuracy measures are:

### Normal event detection accuracy

\[
\frac{TN}{TN + FP}
=
\frac{26286}{26286 + 4363}
=
0.8576
\]

**Normal accuracy = 85.76%**

### Anomaly event detection accuracy

\[
\frac{TP}{TP + FN}
=
\frac{106}{106 + 35}
=
0.7518
\]

**Anomaly accuracy = 75.18%**

Therefore, both assignment targets are satisfied:

- Normal accuracy >= 80%: **Yes**
- Anomaly accuracy >= 75%: **Yes**

## Figure

The figure below shows the nitrate time series with the points classified as anomalies highlighted:

![Detected nitrate anomalies](hw2_anomaly_plot.png)

## Design choices

- I used an **upper-tail, one-sided threshold**, because the assignment explicitly permits focusing on high nitrate values.
- The current/new point is included in the current window before the percentile is calculated, exactly as described in the assignment.
- The first window is handled by labeling all `W` observations with the first threshold.
- Later windows label only the newly added point.
- Percentiles are calculated with `np.percentile(..., method="linear")`.
- The supplied cleaned file contains no missing values in `NO3N`, so no additional NaN removal or imputation is performed.
- The CSV uses the column name `NO3N` for nitrate and `Student_Flag` for the provided ground truth.

## Running the program

Place the Python script and CSV in the same folder, then run:

```bash
python hw2_sliding_window_anomaly_detection.py AG_NO3_fill_cells_remove_NAN.csv
```

If your downloaded file has a slightly different filename, pass that filename instead.

The script prints the metrics and creates:

- `hw2_predictions.csv`
- `hw2_anomaly_plot.png`
