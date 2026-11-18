"""PathSnap FastAPI backend (Laptop 1).

Run from the repo root:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
Trips are read from PATHSNAP_EVAL_DIR (default: data/evaluator_test), one folder per trip.
"""
import logging, os, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")
from api.pipeline import ROOT, FinalModel  # noqa: E402

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "evaluator_test"))
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL = FinalModel()


def trip_dirs():
    if not EVAL_DIR.is_dir():
        return {}
    return {d.name: d for d in sorted(EVAL_DIR.iterdir()) if d.is_dir() and (d / f"{d.name}.track").exists()}


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "trips_dir": str(EVAL_DIR), "trips": len(trip_dirs())}


@app.get("/trips")
def trips():
    out = []
    for tid, d in trip_dirs().items():
        with open(d / f"{tid}.track") as fh:
            n = sum(1 for line in fh if line.strip())
        out.append({"trip_id": tid, "gps_points": n})
    log.info("GET /trips -> %d trips from %s", len(out), EVAL_DIR)
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip().zfill(8)
    dirs = trip_dirs()
    if tid not in dirs:
        raise HTTPException(404, f"trip {tid} not found in {EVAL_DIR}")
    log.info("POST /predict  trip_id=%s  -> running full pipeline", tid)
    try:
        return MODEL.run(dirs[tid])
    except Exception as e:  # keep the server up, show the error in the log and the UI
        log.exception("trip %s failed", tid)
        raise HTTPException(500, f"pipeline failed for {tid}: {e}")
