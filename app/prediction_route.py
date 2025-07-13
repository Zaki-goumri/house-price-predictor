from joblib.memory import filter_args
from pydantic import BaseModel
from fastapi import APIRouter
import pandas as pd
import joblib


class HouseFearutes(BaseModel):
    data: dict


router = APIRouter()
model_artifacts = joblib.load("models/house_price_predictor.joblib")
model = model_artifacts["model"]
scaler = model_artifacts["scaler"]
columns = model_artifacts["columns"]


@router.post("/predict")
def predict(features: HouseFearutes):
    df = pd.DataFrame([features.data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    x_scaled = scaler.transform(df)
    pred = model.predict(x_scaled)[0]
    return {"predicted_price": pred}
