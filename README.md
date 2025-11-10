
# Ton/h Predictor API (FastAPI)

Predict Ton/h from Fe, FeO, and Recovery using your trained Excel model.


### Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### API Endpoints
- `GET /health` – Check service status and valid input ranges.
- `POST /predict` – Input JSON: `{"Fe": 47.2936, "FeO": 16.905, "Recovery": 0.715497}`

### Docker
```bash
docker build -t tonh-api .
docker run -p 8000:8000 tonh-api
```
