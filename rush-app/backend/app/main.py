from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import RushFeatures, PredictionResponse
from .model_service import predict_all, load_models

app = FastAPI(title="Hockey Rush Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: RushFeatures):
    return predict_all(features.model_dump())