# Amrit's PathSnap feature-engineering starter files

These files implement the candidate-generation and geometric/motion feature layer.
They deliberately do not implement reference-route/GPS alignment or training labels:
that alignment must be established and validated against the hackathon data before
labels are generated. Do not assume GPS row i corresponds to route row i.

## Dependencies

Install (or add to the team's shared requirements):
    pip install numpy pandas scipy shapely pyproj pytest

## Expected input formats

- `track`: columns `[longitude, latitude, timestamp_seconds]`
- `nodes`: columns `[longitude, latitude]`
- `arcs`: columns `[from_node_id, to_node_id]`

The arc ID is the zero-based row index in the `.arcs` file.

## Example

```python
import pandas as pd
from features.candidate_generation import build_road_network_index, generate_candidates
from features.feature_extraction import build_candidate_feature_table

track = pd.read_csv("00000001.track", sep=r"\s+", header=None,
                    names=["longitude", "latitude", "timestamp"])
nodes = pd.read_csv("00000001.nodes", sep=r"\s+", header=None,
                    names=["longitude", "latitude"])
arcs = pd.read_csv("00000001.arcs", sep=r"\s+", header=None,
                   names=["from_node_id", "to_node_id"])

index = build_road_network_index(nodes, arcs)
candidates = generate_candidates(track, index, search_radius_m=40, max_candidates=8)
feature_table = build_candidate_feature_table(track, candidates, index.transformer)

print(feature_table.head())
```

## Important implementation notes

- Distances are calculated in a local UTM projection (metres), not degrees.
- The road graph is directed; opposite-direction arcs remain distinct.
- Candidate generation returns zero rows for an observation when no road is
  within the configured search radius. Track this explicitly; do not silently
  pretend every GPS point has a candidate.
- `candidate_coverage` can evaluate candidate recall only after a defensible
  mapping from GPS indices to reference route arc IDs has been established.
- The reference `.route` file must not be used as an input feature.
- These are initial implementation files. Validate runtime and candidate recall
  on actual trips and tune the search radius/candidate cap empirically.
