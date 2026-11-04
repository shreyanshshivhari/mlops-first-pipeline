import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel, Field
from mlflow import MlflowClient
from contextlib import asynccontextmanager
from typing import Callable

# app = FastAPI(title="Iris Model API")

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:6001"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URL = "models:/iris-classifier@champion"

def load_mlflow_model():
    mlflow.set_tracking_url(MLFLOW_TRACKING_URI)
    return mlflow.sklean.load_model(MODEL_URL)



class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

def create_app(model_loader:Callable=load_mlflow_model,
               version_loader:Callable=None):
    if version_loader is None:
        def version_loader():
            client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            return client.get_model_version_by_alias(name="iris-classifier", alias="champion")
        return champion.version
    #lifepsan function excecutes when application starts......
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        #Load the model once during the application startup
        #injected loader can be real mlflow or a test model
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(f"Failed to load champion model..\n {ex}")
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        champion=client.get_model_version_by_alias(name="iris-classifier", alias="champion")
        app.state.model_version=champion.version
        print(f"loaded iris-classifier version {app.state.model_version}")
        yield   

    app=FastAPI(title="Iris Classification API",description="API for classifying iris flowers",version="3.0.0", lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "API is ready to serve..."}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: IrisRequest,request:Request):
    model = request.app.state.model
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "class_name": [
            "setosa",
            "versicolor",
            "virginica"
        ][prediction]
    }