# Dementia Risk Assessment for Atrial Fibrillation

Web application for 5-year dementia risk prediction in AF patients using XGBoost-Cox survival model.

## Prerequisites

- Python 3.8+
- Node.js 16+

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend runs on `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on `http://localhost:3000`

## Structure

```
Clinical App/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── App.css
│   └── package.json
├── xgb_cox_model.json
├── baseline_hazard.json
├── feature_manifest.json
└── app_metadata.json
```

## Deployment

For production deployment, set the `REACT_APP_API_URL` environment variable to your backend URL.