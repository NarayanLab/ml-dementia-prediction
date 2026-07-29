# Dementia Risk Assessment for Atrial Fibrillation

Web application for 5-year dementia risk prediction in AF patients, using a
22-feature XGBoost-Cox survival model (held-out C-index 0.822).

Live demo: https://ml-dementia-prediction-1.onrender.com/

## Prerequisites

- Python 3.11 (see `backend/.python-version`)
- Node.js 16+

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend runs on `http://localhost:8000`. Check `GET /health` to confirm which
model got loaded — it reports the model filename, feature count, `S0_tstar`, and
the risk-threshold band.

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on `http://localhost:3000`. On Windows, `start.bat` launches both.

## Tests

```bash
cd backend && python test_backend.py     # model wiring + behavioural checks
cd frontend && npm test                  # form renders the required inputs
```

`test_backend.py` guards the mistakes that have actually happened here: a booster
with the wrong feature count, column order read from the importance-ranking file
instead of the model, and features silently left at their population mean rather
than coming from the form.

## Structure

```
Clinical App/
├── backend/
│   ├── main.py               FastAPI app; MODEL_DIR points at ../model
│   ├── test_backend.py
│   └── requirements.txt
├── frontend/
│   ├── src/App.tsx           the 22-input form
│   └── package.json
├── model/                    served model + its metadata (see model/README.md)
│   ├── xgb_cox_22feature.json
│   ├── meta.json
│   ├── feature_means.json
│   ├── baseline.json
│   ├── dca_band.json
│   ├── training_meta.json
│   └── feature_importance_ranking.json
└── start.bat
```

The 77-feature parent model and the study data dictionary are intentionally not
in version control; see `model/README.md` for provenance and for the encodings
the API expects.

## Model inputs

22 features, all supplied by the form — nothing is imputed at request time except
Calcium when the clinician marks it unavailable.

Two details worth knowing before reading a prediction:

- **QRS is axis in degrees** (roughly -80 to +195), not duration in ms.
- `Calcium_missing` and `HCT_missing` are the only missingness flags in the model,
  which is why exactly those two labs have availability toggles.

## Deployment

Both services are hosted on Render and configured through its dashboard (there is
no `render.yaml` in this repo).

- **Frontend** — static site built from `frontend/`. Set `REACT_APP_API_URL` to
  the backend service URL.
- **Backend** — Python 3.11 web service from `backend/`. The start command must
  bind Render's injected port, e.g.
  `uvicorn main:app --host 0.0.0.0 --port $PORT`. The `__main__` block in
  `main.py` hardcodes 8000 and is for local use only.
- The backend's CORS allowlist in `main.py` must contain the frontend's origin.

`model/` must be deployed with the backend — the app will not start without it.
