# RakshaMitra ML Engine — Vercel-ready layout

Same XGBoost + SHAP + VADER pipeline as before, restructured so the same
folder works for **both** local development and a direct `vercel deploy` —
you don't need to maintain two copies.

```
ml-engine-vercel/
├── api/
│   ├── index.py            <- FastAPI app (Vercel's recognized entrypoint)
│   ├── risk_pipeline.py
│   ├── explain_engine.py
│   ├── sentiment_engine.py
│   └── model/*.joblib
├── requirements.txt
├── vercel.json              <- maxDuration: 30 (cold starts with xgboost/shap take longer than the 10s default)
├── data_generator.py         <- only needed if you ever retrain
└── train_model.py
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows cmd
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cd api
uvicorn index:app --reload --port 8000
```

Verify at `http://127.0.0.1:8000/docs`. Set the frontend's `.env` to
`ML_API_URL=http://127.0.0.1:8000` exactly as before — nothing else changes
for local dev.

## Deploy to Vercel

```bash
npm i -g vercel        # one-time, if you don't have the CLI
cd ml-engine-vercel
vercel login
vercel --prod
```

Vercel auto-detects `api/index.py` as a Python Function (zero config needed
beyond `vercel.json`). Once it finishes, it prints a URL like
`https://your-project.vercel.app` — that's your backend's public base URL.

**Test it:** `https://your-project.vercel.app/docs` should show the same
Swagger UI as local. Then point the frontend's `ML_API_URL` at
`https://your-project.vercel.app` (no trailing slash) instead of
`127.0.0.1:8000`.

## Cold starts — read this before your demo

The first request after the function has been idle can take several
seconds while Python re-imports pandas/xgboost/shap and reloads the model.
Hit `/health` yourself right before you go on stage/in front of judges to
warm it up — don't let the judges' first check-in be the cold one.
