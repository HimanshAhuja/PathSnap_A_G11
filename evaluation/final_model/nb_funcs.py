META = ["trip_id", "gps_index", "candidate_arc_id"]
TARGET = "label"
BASE_FEATURES = ["distance", "bearing_difference", "speed", "heading_change", "segment_length"]
RENAME = {"distance_m": "distance", "bearing_difference_deg": "bearing_difference",
          "speed_mps": "speed", "heading_change_deg": "heading_change", "segment_length_m": "segment_length"}

def load_table(path):
    df = pd.read_csv(path, dtype={"trip_id": str}).rename(columns=RENAME)
    miss = [c for c in META + BASE_FEATURES + [TARGET] if c not in df.columns]
    assert not miss, f"{path}: missing columns {miss}"
    df["trip_id"] = df["trip_id"].str.zfill(8)
    df[BASE_FEATURES] = df[BASE_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df[TARGET] = df[TARGET].astype(int)
    df["gps_index"] = df["gps_index"].astype(int)
    df["candidate_arc_id"] = df["candidate_arc_id"].astype(int)
    return df.sort_values(["trip_id", "gps_index", "candidate_arc_id"]).reset_index(drop=True)


def load_raw_split(name):
    out = {}
    for d in sorted(glob.glob(os.path.join(split_dir(name), "*"))):
        tid = os.path.basename(d)
        arcs = np.loadtxt(os.path.join(d, f"{tid}.arcs"), dtype=np.int64, ndmin=2)
        nodes = np.loadtxt(os.path.join(d, f"{tid}.nodes"), dtype=float, ndmin=2)
        track = np.loadtxt(os.path.join(d, f"{tid}.track"), dtype=float, ndmin=2)
        route = np.loadtxt(os.path.join(d, f"{tid}.route"), dtype=np.int64, ndmin=1)
        lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])          # per-trip local projection (metres)
        kx = 6371000.0 * np.pi / 180 * np.cos(np.radians(lat0)); ky = 6371000.0 * np.pi / 180
        P = np.c_[(track[:, 0] - lon0) * kx, (track[:, 1] - lat0) * ky]
        N = np.c_[(nodes[:, 0] - lon0) * kx, (nodes[:, 1] - lat0) * ky]
        out[tid.zfill(8)] = dict(frm=arcs[:, 0], to=arcs[:, 1], t=track[:, 2], n_gps=len(track), route=route, P=P, N=N)
    return out


def add_features(df, raw):
    df = df.copy()
    keys = ["trip_id", "gps_index"]
    g = df.groupby(keys, sort=False)
    df["n_candidates"] = g["distance"].transform("size")
    df["dist_rank"] = g["distance"].rank(method="first")
    dmin = g["distance"].transform("min")
    df["dist_minus_min"] = df["distance"] - dmin
    df["dist_ratio_min"] = (df["distance"] + 1.0) / (dmin + 1.0)
    w = np.exp(-df["dist_minus_min"] / 10.0)
    df["dist_softmax"] = w / w.groupby([df.trip_id, df.gps_index]).transform("sum")
    df["bear_rank"] = g["bearing_difference"].rank(method="min")
    df["bear_minus_min"] = df["bearing_difference"] - g["bearing_difference"].transform("min")
    df["is_nearest"] = (df["dist_rank"] == 1).astype(int)

    # neighbouring observed GPS points (points without candidates are skipped)
    pts = df[keys].drop_duplicates().sort_values(keys)
    pts["prev_gps"] = pts.groupby("trip_id")["gps_index"].shift(1)
    pts["next_gps"] = pts.groupby("trip_id")["gps_index"].shift(-1)
    nearest = df.loc[df.is_nearest == 1, keys + ["candidate_arc_id"]].rename(columns={"candidate_arc_id": "nearest_arc"})
    pts = pts.merge(nearest.rename(columns={"gps_index": "prev_gps", "nearest_arc": "prev_nearest"}), on=["trip_id", "prev_gps"], how="left")
    pts = pts.merge(nearest.rename(columns={"gps_index": "next_gps", "nearest_arc": "next_nearest"}), on=["trip_id", "next_gps"], how="left")
    df = df.merge(pts, on=keys, how="left")

    cand = df[keys + ["candidate_arc_id"]].assign(_hit=1)
    df = df.merge(cand.rename(columns={"gps_index": "prev_gps", "_hit": "in_prev_cands"}),
                  on=["trip_id", "prev_gps", "candidate_arc_id"], how="left")
    df = df.merge(cand.rename(columns={"gps_index": "next_gps", "_hit": "in_next_cands"}),
                  on=["trip_id", "next_gps", "candidate_arc_id"], how="left")
    df[["in_prev_cands", "in_next_cands"]] = df[["in_prev_cands", "in_next_cands"]].fillna(0)
    df["is_prev_nearest"] = (df.candidate_arc_id == df.prev_nearest).astype(int)
    df["is_next_nearest"] = (df.candidate_arc_id == df.next_nearest).astype(int)
    df["gps_step_prev"] = df.gps_index - df.prev_gps
    df["gps_step_next"] = df.next_gps - df.gps_index

    # road connectivity: arc follows the previous nearest arc / leads into the next nearest arc
    from_prev = np.zeros(len(df), dtype=int)
    to_next = np.zeros(len(df), dtype=int)
    for tid, idx in df.groupby("trip_id").indices.items():
        frm, to = raw[tid]["frm"], raw[tid]["to"]
        a = df.candidate_arc_id.values[idx]
        pn, nn = df.prev_nearest.values[idx], df.next_nearest.values[idx]
        ok_p, ok_n = ~np.isnan(pn), ~np.isnan(nn)
        from_prev[idx[ok_p]] = ((to[pn[ok_p].astype(int)] == frm[a[ok_p]]) | (pn[ok_p].astype(int) == a[ok_p])).astype(int)
        to_next[idx[ok_n]] = ((to[a[ok_n]] == frm[nn[ok_n].astype(int)]) | (nn[ok_n].astype(int) == a[ok_n])).astype(int)
    df["from_prev_nearest"], df["to_next_nearest"] = from_prev, to_next
    df = df.drop(columns=["prev_gps", "next_gps", "prev_nearest", "next_nearest"])
    return df.sort_values(["trip_id", "gps_index", "candidate_arc_id"]).reset_index(drop=True)

def window_vectors(P, min_disp=25.0, kmax=20):
    # displacement over the smallest symmetric window (i-k .. i+k) that moved >= min_disp metres
    n = len(P); V = np.zeros((n, 2)); done = np.zeros(n, bool); idx = np.arange(n)
    for k in range(1, kmax + 1):
        a, b = np.clip(idx - k, 0, n - 1), np.clip(idx + k, 0, n - 1); v = P[b] - P[a]
        ok = (~done) & (np.hypot(v[:, 0], v[:, 1]) >= min_disp); V[ok] = v[ok]; done |= ok
    a, b = np.clip(idx - kmax, 0, n - 1), np.clip(idx + kmax, 0, n - 1); V[~done] = (P[b] - P[a])[~done]
    return V

def add_geometry_features(df, raw):
    cols = ["along_motion_m", "win_bearing_diff", "window_disp_m", "proj_t", "beyond_end_m", "before_start_m",
            "signed_offset_m", "pos_m"]
    out = {c: np.full(len(df), np.nan) for c in cols}
    for tid, idx in df.groupby("trip_id").indices.items():
        r = raw[tid]; V = window_vectors(r["P"])
        gi = df.gps_index.values[idx]; a = df.candidate_arc_id.values[idx]
        A, B = r["N"][r["frm"][a]], r["N"][r["to"][a]]; av = B - A
        L = np.maximum(np.hypot(av[:, 0], av[:, 1]), 1e-6); u = av / L[:, None]
        v = V[gi]; vn = np.hypot(v[:, 0], v[:, 1]); p = r["P"][gi]
        out["along_motion_m"][idx] = (v * u).sum(1)
        out["window_disp_m"][idx] = vn
        out["win_bearing_diff"][idx] = np.degrees(np.arccos(np.clip((v * u).sum(1) / np.maximum(vn, 1e-6), -1, 1)))
        t = ((p - A) * u).sum(1) / L
        out["proj_t"][idx] = t
        out["pos_m"][idx] = t * L                                   # position along the arc (used by the HMM)
        out["beyond_end_m"][idx] = np.maximum(0, (t - 1) * L)
        out["before_start_m"][idx] = np.maximum(0, -t * L)
        out["signed_offset_m"][idx] = u[:, 0] * (p - A)[:, 1] - u[:, 1] * (p - A)[:, 0]
    df = df.copy()
    for c in cols:
        df[c] = out[c]
    df["along_motion_norm"] = df.along_motion_m / np.maximum(df.window_disp_m, 1.0)
    df["span_pen"] = df.beyond_end_m + df.before_start_m
    g = df.groupby(["trip_id", "gps_index"], sort=False)
    df["win_bd_rank"] = g["win_bearing_diff"].rank(method="min")
    df["span_pen_rank"] = g["span_pen"].rank(method="min")
    return df


FEATURES = BASE_FEATURES + ["n_candidates", "dist_rank", "dist_minus_min", "dist_ratio_min", "dist_softmax",
                            "bear_rank", "bear_minus_min", "is_nearest", "in_prev_cands", "in_next_cands",
                            "is_prev_nearest", "is_next_nearest", "from_prev_nearest", "to_next_nearest",
                            "gps_step_prev", "gps_step_next",
                            "along_motion_m", "along_motion_norm", "win_bearing_diff", "window_disp_m", "win_bd_rank",
                            "proj_t", "beyond_end_m", "before_start_m", "span_pen", "span_pen_rank", "signed_offset_m"]

POINT = ["trip_id", "gps_index"]

def top1_choice(df, score):
    d = df[POINT + ["candidate_arc_id", TARGET]].copy()
    d["score"] = np.asarray(score)
    idx = d.groupby(POINT, sort=False)["score"].idxmax()
    return d.loc[idx].sort_values(POINT).reset_index(drop=True)

def route_continuity(choice, split):
    ok = total = 0
    for tid, g in choice.groupby("trip_id"):
        a = g.sort_values("gps_index")["candidate_arc_id"].to_numpy()
        frm, to = RAW[split][tid]["frm"], RAW[split][tid]["to"]
        ch = a[1:] != a[:-1]
        ok += int((to[a[:-1][ch]] == frm[a[1:][ch]]).sum()); total += int(ch.sum())
    return ok / total if total else 1.0

def compute_metrics(df, proba, split=None):
    y = df[TARGET].to_numpy(); p = np.clip(np.asarray(proba), 1e-7, 1 - 1e-7); yhat = (p >= 0.5).astype(int)
    ch = top1_choice(df, p)
    m = {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
         "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
         "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
         "log_loss": log_loss(y, p, labels=[0, 1]), "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
         "top1_accuracy": ch[TARGET].mean()}
    if split is not None:
        m["route_continuity"] = route_continuity(ch, split)
    return m


def prepare_trips(df, proba, split):
    """Per trip: list of per-observation candidate arrays plus precomputed relation codes."""
    d = df[POINT + ["candidate_arc_id", TARGET, "pos_m"]].copy(); d["p"] = np.asarray(proba)
    trips = {}
    for tid, g in d.groupby("trip_id", sort=False):
        frm, to, tt = RAW[split][tid]["frm"], RAW[split][tid]["to"], RAW[split][tid]["t"]
        succ_to = defaultdict(set)                      # node -> nodes reachable with one arc
        for f_, t_ in zip(frm, to):
            succ_to[f_].add(t_)
        obs = [(gi, gg.candidate_arc_id.to_numpy(), gg.p.to_numpy(), gg[TARGET].to_numpy(), gg.pos_m.to_numpy())
               for gi, gg in g.groupby("gps_index", sort=True)]
        codes, dts, backs = [None], [0.0], [None]
        for k in range(1, len(obs)):
            a_prev, a_cur = obs[k - 1][1], obs[k][1]
            same = a_prev[:, None] == a_cur[None, :]
            nxt = to[a_prev][:, None] == frm[a_cur][None, :]
            two = np.array([[frm[b] in succ_to[to[a]] for b in a_cur] for a in a_prev])
            uturn = (to[a_prev][:, None] == frm[a_cur][None, :]) & (frm[a_prev][:, None] == to[a_cur][None, :])
            c = np.full(same.shape, 3, dtype=np.int8); c[two] = 2; c[nxt] = 1; c[uturn] = 4; c[same] = 0
            back = np.where(same, np.clip((obs[k - 1][4][:, None] - obs[k][4][None, :] - 5.0) / 10.0, 0, 3), 0.0)
            codes.append(c); backs.append(back); dts.append(float(tt[obs[k][0]] - tt[obs[k - 1][0]]))
        trips[tid] = dict(obs=obs, codes=codes, dts=dts, backs=backs)
    return trips

def viterbi_trip(trip, hp):
    pen = np.array([0.0, hp["next_pen"], hp["two_pen"], hp["far_pen"], hp["uturn_pen"]])
    obs = trip["obs"]
    delta = hp["w"] * np.log(np.clip(obs[0][2], 1e-6, 1)); back = [None]
    for k in range(1, len(obs)):
        dt = trip["dts"][k]
        if dt > hp["gap_reset"]:
            A = np.zeros(trip["codes"][k].shape)
        else:
            A = (pen[trip["codes"][k]] + hp["back_pen"] * trip["backs"][k]) / (1.0 + max(0.0, dt - 1.0) / hp["tau"])
        scores = delta[:, None] + A
        bp = scores.argmax(axis=0)
        delta = scores[bp, np.arange(scores.shape[1])] + hp["w"] * np.log(np.clip(obs[k][2], 1e-6, 1))
        back.append(bp)
    path = [int(delta.argmax())]
    for k in range(len(obs) - 1, 0, -1):
        path.append(int(back[k][path[-1]]))
    path.reverse()
    return [(obs[k][0], int(obs[k][1][s]), int(obs[k][3][s])) for k, s in enumerate(path)]

def decode(trips, hp):
    rows = [(tid, gi, arc, lab) for tid, tr in trips.items() for gi, arc, lab in viterbi_trip(tr, hp)]
    return pd.DataFrame(rows, columns=["trip_id", "gps_index", "candidate_arc_id", TARGET])

