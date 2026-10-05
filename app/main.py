"""FastAPI service that serves the Iris classifier."""
import os
import socket
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.train import MODEL_PATH, train

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        train()
    state["bundle"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Iris ML API", version="1.0.0", lifespan=lifespan)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, description="cm")
    sepal_width: float = Field(..., gt=0, lt=20, description="cm")
    petal_length: float = Field(..., gt=0, lt=20, description="cm")
    petal_width: float = Field(..., gt=0, lt=20, description="cm")


@app.get("/")
def root():
    return {"service": "iris-ml-api", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "bundle" in state,
        "host": os.getenv("HOSTNAME", socket.gethostname()),
    }


@app.post("/predict")
def predict(features: IrisFeatures):
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    model = bundle["model"]
    idx = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    return {
        "prediction": bundle["classes"][idx],
        "confidence": round(float(proba[idx]), 4),
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(bundle["classes"], proba)
        },
    }
