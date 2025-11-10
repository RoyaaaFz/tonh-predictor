
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import uvicorn
import io
import os
import threading

APP_TITLE = "Ton/h Predictor API"
DESCRIPTION = "FastAPI service with a clean UI that predicts Ton/h from Fe, FeO, and Recovery using trained RF/XGB grids."
VERSION = "1.1.1"

# Ensure folders exist so StaticFiles/Jinja2 don't error
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(title=APP_TITLE, description=DESCRIPTION, version=VERSION)

# Static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

_lock = threading.Lock()

class PredictRequest(BaseModel):
    Fe: float = Field(..., description="Fe value within the trained grid range")
    FeO: float = Field(..., description="FeO value within the trained grid range")
    Recovery: float = Field(..., description="Recovery as fraction (0-1) or percent (0-100)")

class ModelStats(BaseModel):
    p10: float
    mean: float
    p90: float

class PredictResponse(BaseModel):
    RF: ModelStats
    XGB: ModelStats
    overall_range: Tuple[float, float] = Field(..., description="Conservative range across models: [min p10, max p90]")

_rf_grids: Dict = {}
_xgb_grids: Dict = {}
_ranges = {"Fe": (None, None), "FeO": (None, None)}

def _load_grids(xl: pd.ExcelFile, model_prefix: str):
    grids = {}
    for r in [10, 50, 90]:
        for stat in ["p10", "mean", "p90"]:
            sheet = f"{model_prefix}_R{r}_{stat}"
            df = xl.parse(sheet)
            fe_values = df.columns[1:].astype(float)
            feo_values = df.iloc[:, 0].astype(float).values
            grid = df.iloc[:, 1:].astype(float).values
            grids[(r, stat)] = {"Fe": fe_values, "FeO": feo_values, "Z": grid}
    return grids

def _common_ranges(grids: Dict):
    fe_mins, fe_maxs, feo_mins, feo_maxs = [], [], [], []
    for v in grids.values():
        fe_mins.append(v["Fe"].min()); fe_maxs.append(v["Fe"].max())
        feo_mins.append(v["FeO"].min()); feo_maxs.append(v["FeO"].max())
    return (max(fe_mins), min(fe_maxs)), (max(feo_mins), min(feo_maxs))

def _bilinear_interpolate(x, y, xs, ys, Z):
    x = float(x); y = float(y)
    x = np.clip(x, xs.min(), xs.max())
    y = np.clip(y, ys.min(), ys.max())
    ix = np.searchsorted(xs, x) - 1
    iy = np.searchsorted(ys, y) - 1
    ix = np.clip(ix, 0, len(xs) - 2)
    iy = np.clip(iy, 0, len(ys) - 2)
    x1, x2 = xs[ix], xs[ix + 1]
    y1, y2 = ys[iy], ys[iy + 1]
    Q11 = Z[iy, ix]; Q21 = Z[iy, ix + 1]
    Q12 = Z[iy + 1, ix]; Q22 = Z[iy + 1, ix + 1]
    wx = 0.0 if x2 == x1 else (x - x1) / (x2 - x1)
    wy = 0.0 if y2 == y1 else (y - y1) / (y2 - y1)
    return float(Q11 * (1 - wx) * (1 - wy) +
                 Q21 * wx * (1 - wy) +
                 Q12 * (1 - wx) * wy +
                 Q22 * wx * wy)

def _interp_recovery(rec, model_grids, stat, Fe, FeO):
    anchors = np.array([0.10, 0.50, 0.90])
    r = float(rec)
    if r > 1.0:
        r = r / 100.0
    r = float(np.clip(r, anchors.min(), anchors.max()))
    vals = []
    for a, r_label in zip(anchors, [10, 50, 90]):
        g = model_grids[(r_label, stat)]
        vals.append(_bilinear_interpolate(Fe, FeO, g["Fe"], g["FeO"], g["Z"]))
    vals = np.array(vals)
    return float(np.interp(r, anchors, vals))

def _predict_all(Fe: float, FeO: float, Recovery: float):
    out = {}
    for name, grids in [("RF", _rf_grids), ("XGB", _xgb_grids)]:
        p10 = _interp_recovery(Recovery, grids, "p10", Fe, FeO)
        mean = _interp_recovery(Recovery, grids, "mean", Fe, FeO)
        p90 = _interp_recovery(Recovery, grids, "p90", Fe, FeO)
        out[name] = {"p10": p10, "mean": mean, "p90": p90}
    overall = (min(out["RF"]["p10"], out["XGB"]["p10"]),
               max(out["RF"]["p90"], out["XGB"]["p90"]))
    return out, overall

def _load_from_excel_bytes(data: bytes):
    xl = pd.ExcelFile(io.BytesIO(data))
    rf = _load_grids(xl, "RF")
    xgb = _load_grids(xl, "XGB")
    (fe_min, fe_max), (feo_min, feo_max) = _common_ranges(rf)
    return rf, xgb, {"Fe": (fe_min, fe_max), "FeO": (feo_min, feo_max)}

# def _try_initial_load():
#     env_path = os.getenv("MODEL_XLSX", r"C:\Users\8888\Desktop\Fe_Feo_Recovery_Ton_h_Project\Model\outputs_RF_XGB_DT_STACK_Tonh_New.xlsx")
#     if not os.path.exists(env_path):
#         raise FileNotFoundError(f"Excel model file not found at '{env_path}'.")
#     with open(env_path, "rb") as f:
#         data = f.read()
#     return _load_from_excel_bytes(data)

def _try_initial_load():
    # Prefer env var, then local bundled files inside the container/repo
    candidates = []
    env_path = os.getenv("MODEL_XLSX")
    if env_path:
        candidates.append(env_path)
    candidates += ["./model.xlsx", "/app/model.xlsx"]

    for p in candidates:
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                data = f.read()
            return _load_from_excel_bytes(data)

    raise FileNotFoundError(
        f"Excel model file not found. Tried: {candidates}. "
        f"Set MODEL_XLSX to a valid path or include model.xlsx at the repo root."
    )


with _lock:
    _rf_grids, _xgb_grids, _ranges = _try_initial_load()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "ranges": _ranges})

@app.get("/health")
def health():
    return {"status": "ok", "Fe_range": _ranges["Fe"], "FeO_range": _ranges["FeO"]}

@app.get("/ranges")
def ranges():
    return _ranges

class _PredictIn(BaseModel):
    Fe: float
    FeO: float
    Recovery: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: _PredictIn):
    (fe_min, fe_max) = _ranges["Fe"]
    (feo_min, feo_max) = _ranges["FeO"]
    Fe = float(np.clip(req.Fe, fe_min, fe_max))
    FeO = float(np.clip(req.FeO, feo_min, feo_max))
    preds, overall = _predict_all(Fe, FeO, req.Recovery)
    return PredictResponse(
        RF=ModelStats(**preds["RF"]),
        XGB=ModelStats(**preds["XGB"]),
        overall_range=overall
    )

@app.post("/reload")
async def reload_excel(file: UploadFile = File(...)):
    data = await file.read()
    try:
        rf, xgb, ranges = _load_from_excel_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load Excel: {e}")
    with _lock:
        global _rf_grids, _xgb_grids, _ranges
        _rf_grids, _xgb_grids, _ranges = rf, xgb, ranges
    return {"status": "reloaded", "Fe_range": _ranges["Fe"], "FeO_range": _ranges["FeO"]}


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.getenv("PORT", "8000"))  # Railway sets PORT; fallback for local dev
    uvicorn.run("main:app", host="0.0.0.0", port=port)
