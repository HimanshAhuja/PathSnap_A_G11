# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus) — Team A_G11

Reconstructs vehicle routes by matching noisy GPS observations to a road network:
candidate generation → feature engineering → candidate classifier → HMM/Viterbi decoding →
FastAPI backend (Laptop 1) → evaluator UI (Laptop 2).

## Team and ownership

| Member | SRN | Ownership |
|---|---|---|
| Aaron Sijo | PES2UG24CS012 | Dataset exploration & preprocessing (`preprocessing/`) |
| Amrit Raj Anmol | PES2UG24CS056 | Candidate generation & feature engineering (`features/`) |
| Abhishek Kumar Singh | PES2UG24CS026 | Model selection & training (`models/`) |
| Himansh Ahuja | PES2UG24CS923 | Integration, validation & evaluation (`evaluation/`, project setup) |

## Project structure

```
PathSnap_A_G11/
├── config.py                  # data paths, agreed schema, random seed (import from here)
├── data/                      # put train/ and student_test/ here (not committed)
├── preprocessing/
│   ├── geo_utils.py           # shared: per-trip metre projection, bearings, point-to-segment
│   ├── load_data.py           # load_trip(split, trip_id)
│   ├── clean_data.py          # clean_track(track)
│   └── dataset_statistics.py
├── features/
│   ├── candidate_generation.py
│   ├── feature_extraction.py
│   └── build_training_data.py # build_trip_features(trip)
├── models/
│   ├── train_model.py
│   ├── evaluate_models.py
│   └── best_model.pkl
├── evaluation/
│   ├── validate_pipeline.py   # end-to-end run of one trip + integrity checks
│   ├── generate_metrics.py    # Top-1 accuracy, candidate miss rate, route continuity
│   ├── generate_plots.py      # model comparison, feature distributions, importance
│   └── plot_trip_map.py       # Folium map: roads, GPS, reference route, candidates
├── results/                   # model_comparison.csv, logs/, plots/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Data location: the code looks for a folder containing `train/` and `student_test/`, in this order:
1. the `PATHSNAP_DATA` environment variable, e.g. `export PATHSNAP_DATA="/path/to/Student folder"`
2. `data/` inside the repo
3. `../Student folder` (repo cloned next to the supplied materials)

The raw data is not committed. Evaluator-testing data is never stored in this repo or used for development.

## Data format (verified against the files)

| File | Columns (whitespace-separated, no header) | ID convention |
|---|---|---|
| `.track` | longitude, latitude, timestamp (s) | row number = `gps_index` |
| `.nodes` | longitude, latitude | row number = node id (0-based) |
| `.arcs` | from_node, to_node (directed) | row number = arc id (0-based) |
| `.route` | arc_id | chronological reference arcs |

Trips are located in different parts of the world, so distances use a **per-trip local projection**
(`preprocessing/geo_utils.py`): `x = R·Δlon·cos(lat0)`, `y = R·Δlat`, centred on the trip's median
GPS position. The error is far below GPS noise at trip scale (tens of km).

## Interfaces (agreed contract)

| Function | Returns |
|---|---|
| `preprocessing.load_data.load_trip(split, trip_id)` | `dict(trip_id, split, track[lon,lat,t], nodes[lon,lat], arcs[from_node,to_node], route[arc_id] or None)` |
| `preprocessing.clean_data.clean_track(track)` | `(clean_track_df with gps_index column, report_dict)` |
| `features.build_training_data.build_trip_features(trip)` | DataFrame, one row per (GPS point, candidate arc), columns below |
| `models/best_model.pkl` (joblib) | fitted sklearn Pipeline with `predict_proba` (written by `models/train_model.py`) |

Feature table (minimum schema, defined in `config.py`):

| Column | Meaning |
|---|---|
| `trip_id`, `gps_index`, `candidate_arc_id` | identifiers — metadata, never model inputs |
| `distance` | GPS-to-arc distance, metres |
| `bearing_difference` | wrapped movement-vs-arc bearing difference, degrees [0, 180] |
| `speed` | m/s from neighbouring observations |
| `heading_change` | change in GPS movement heading, degrees |
| `segment_length` | arc length, metres |
| `label` | 1 if the candidate is the aligned reference arc, else 0 |

Extra numeric feature columns are picked up automatically by `config.feature_columns(df)`.

## Running

```bash
# End-to-end validation of one real trip (stages not written yet show as PENDING)
python -m evaluation.validate_pipeline --split train --trip 00000046
python -m evaluation.validate_pipeline --split student_test --trip 00000052

# Validate a saved feature table
python -m evaluation.validate_pipeline --features-file results/training_data.csv

# Metrics table and plots (from real run outputs only)
python -m evaluation.generate_metrics
python -m evaluation.generate_plots models
python -m evaluation.generate_plots features --file results/training_data.csv
python -m evaluation.generate_plots importance

# Interactive Folium map of a trip (optionally with one observation's candidates)
python -m evaluation.plot_trip_map --split train --trip 00000046
python -m evaluation.plot_trip_map --split train --trip 00000046 \
    --features-file results/training_data.csv --gps-index 120
```

Validation logs are written to `results/logs/`, plots and maps to `results/plots/`.

## Validation checks performed

Raw data: finite and in-range coordinates, valid node/arc indices, route arcs exist and are
connected head-to-tail, timestamp ordering and gaps.
Feature table: required schema, no ID columns used as features, no NaN/inf, no duplicate
(trip, gps_index, arc) rows, labels in {0, 1}, at most one positive per GPS point, valid arc
and GPS indices, empty candidate sets, candidate miss rate and label balance.

## Status

- [x] Project structure, config, shared geometry, validation harness, metrics, plots, map
- [ ] Preprocessing — loaders, cleaning, dataset statistics
- [ ] Candidate generation and features
- [ ] Five-model comparison
- [ ] Bayesian optimisation, HMM/Viterbi, FastAPI, UI
