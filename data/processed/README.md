# Processed data (output of preprocessing, candidate generation and feature extraction)

These are the tables the models were trained and tested on. They are produced by
`preprocessing/` (loading, cleaning) and `features/` (candidate generation, feature
extraction, labelling) from the raw `.track/.nodes/.arcs/.route` files.

| File | What it is | Size |
|---|---|---|
| `training_data.csv.gz` | Candidate + feature table for the 80 training trips: 965,369 candidate rows, 183,926 GPS points, one row per (GPS point, candidate arc) | 26 MB (109 MB unzipped) |
| `student_test_data.csv.gz` | Same table for the 10 student-test trips: 103,776 rows, 19,096 GPS points. Used only for the final evaluation, never for training or tuning | 2.8 MB |
| `student_test_alignment_diagnostics.csv` | Per-point label-alignment check for the student-test trips | 0.2 MB |

Columns: `trip_id, gps_index, candidate_arc_id, distance, bearing_difference, speed,
heading_change, segment_length, label` (label = 1 for the true arc of that GPS point).
The extra geometry and context features (windowed direction, projection position,
rank features) are computed from these columns inside `notebooks/PathSnap_full_pipeline.ipynb`.

The training file is gzipped because GitHub rejects files over 100 MB. pandas reads it directly:

```python
import pandas as pd
train = pd.read_csv("data/processed/training_data.csv.gz", dtype={"trip_id": str})
test  = pd.read_csv("data/processed/student_test_data.csv.gz", dtype={"trip_id": str})
```
