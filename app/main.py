from fastapi import FastAPI
from app.prediction_route import router

app = FastAPI(title="AI Prediction API")
app.include_router(router)


@app.get("/")
def main():
    return "server is running"
