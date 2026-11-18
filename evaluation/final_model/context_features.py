def context_features(df, p, split):
    d = df[POINT + ["candidate_arc_id"]].copy(); d["p"] = np.asarray(p)
    frm = np.empty(len(d), dtype=np.int64); to = np.empty(len(d), dtype=np.int64)
    for tid, idx in d.groupby("trip_id").indices.items():
        a = d.candidate_arc_id.values[idx]; frm[idx] = RAW[split][tid]["frm"][a]; to[idx] = RAW[split][tid]["to"][a]
    d["frm"], d["to"] = frm, to
    g = d.groupby(POINT, sort=False)["p"]
    d["p_rank"] = g.rank(ascending=False, method="first")
    d["p_max"] = g.transform("max"); d["p_gap"] = d.p_max - d.p
    second = d[d.p_rank == 2].set_index(POINT)["p"].rename("p_second")
    d = d.merge(second, left_on=POINT, right_index=True, how="left"); d["p_second"] = d.p_second.fillna(0)
    d["p_margin"] = np.where(d.p_rank == 1, d.p - d.p_second, d.p - d.p_max)
    pts = d[POINT].drop_duplicates().sort_values(POINT)
    for k in (1, 2):
        pts[f"prev{k}"] = pts.groupby("trip_id")["gps_index"].shift(k)
        pts[f"next{k}"] = pts.groupby("trip_id")["gps_index"].shift(-k)
    d = d.merge(pts, on=POINT, how="left")
    same = d[["trip_id", "gps_index", "candidate_arc_id", "p"]]
    into = d.groupby(["trip_id", "gps_index", "to"])["p"].max().reset_index()      # best arc ending at node
    outof = d.groupby(["trip_id", "gps_index", "frm"])["p"].max().reset_index()    # best arc starting at node
    for k in (1, 2):
        d = d.merge(same.rename(columns={"gps_index": f"prev{k}", "p": f"prev{k}_same"}), on=["trip_id", f"prev{k}", "candidate_arc_id"], how="left")
        d = d.merge(same.rename(columns={"gps_index": f"next{k}", "p": f"next{k}_same"}), on=["trip_id", f"next{k}", "candidate_arc_id"], how="left")
        d = d.merge(into.rename(columns={"gps_index": f"prev{k}", "to": "frm", "p": f"prev{k}_into"}), on=["trip_id", f"prev{k}", "frm"], how="left")
        d = d.merge(outof.rename(columns={"gps_index": f"next{k}", "frm": "to", "p": f"next{k}_from"}), on=["trip_id", f"next{k}", "to"], how="left")
    cols = ["p", "p_rank", "p_gap", "p_margin"] + [f"{s}{k}_{t}" for k in (1, 2) for s, t in
                                                    [("prev", "same"), ("next", "same"), ("prev", "into"), ("next", "from")]]
    d[cols] = d[cols].fillna(0.0)
    for k in (1, 2):
        d[f"ctx{k}"] = np.maximum(d[f"prev{k}_same"], d[f"prev{k}_into"]) + np.maximum(d[f"next{k}_same"], d[f"next{k}_from"])
    cols += ["ctx1", "ctx2"]
    d = d.sort_values(POINT + ["candidate_arc_id"]).reset_index(drop=True)
    assert len(d) == len(df) and (d.candidate_arc_id.values == df.candidate_arc_id.values).all()
    return d[cols].set_axis(df.index)

