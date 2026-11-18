# 11:30 AM first evaluation — demonstration order

Total ≈ 8 minutes. Himansh drives the laptop and hands over to each owner.

| # | Who | Show | Command / file |
|---|---|---|---|
| 1 | Himansh | Repo structure, ownership, agreed interface/schema | `README.md`, `config.py` |
| 2 | Aaron | One trip's four files and what each contains; trip counts and statistics | `preprocessing/dataset_statistics.py` |
| 3 | Aaron | Loader + data checks on a real trip; how missing values and time gaps are handled | `python -m evaluation.validate_pipeline --split train --trip 00000046` (raw-data checks) |
| 4 | Himansh | Map of the trip: road network, raw GPS, reference route | `python -m evaluation.plot_trip_map --split train --trip 00000046` |
| 5 | Amrit | One GPS observation and its candidate arcs; radius and candidate cap; miss rate | `plot_trip_map ... --features-file ... --gps-index N` |
| 6 | Amrit | Small feature table; how the reference route is aligned to GPS points (and limits) | `features/build_training_data.py` output |
| 7 | Abhishek | Five models, trip-wise GroupKFold, metrics table, provisional choice | `results/model_comparison.csv`, `python -m evaluation.generate_plots models` |
| 8 | Himansh | Full end-to-end run on a student-test trip with checks and Top-1; where logs/artifacts live | `python -m evaluation.validate_pipeline --split student_test --trip 00000052` |

Rules for the demo: only show numbers produced by these commands; label any subset run as a
subset; say clearly which parts are complete and which are pending.

## Likely questions for Himansh

- **How do the modules connect?** Every stage follows the contract in README "Interfaces";
  `validate_pipeline.py` imports each stage by name and runs one trip through all of them.
- **Can you reproduce the run?** Yes — the commands above; seeds in `config.RANDOM_SEED`;
  each validation run writes a JSON log to `results/logs/`.
- **What checks detect invalid candidates/features?** NaN/inf, duplicate candidate rows, arc ids
  outside `.arcs`, GPS indices outside the track, labels not 0/1, more than one positive per
  point, empty candidate sets, ID columns accidentally used as features, candidate miss rate.
- **Why a per-trip projection?** Trips are on different continents, so one UTM zone cannot fit
  all of them; a local flat projection per trip is accurate to well under 1% at trip scale.
- **What is complete vs pending?** The validator prints PENDING for any stage not yet written.
