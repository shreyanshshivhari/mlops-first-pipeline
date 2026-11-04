
import os
import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Callable

# ---------------------------------
# MLFLOW CONFIGURATION
# ---------------------------------

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:6001"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URL = "models:/iris-classifier@champion"


# ---------------------------------
# MODEL LOADER
# ---------------------------------

def load_mlflow_model():
    return mlflow.sklearn.load_model(MODEL_URL)


# ---------------------------------
# REQUEST SCHEMA
# ---------------------------------

class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# ---------------------------------
# APPLICATION FACTORY
# ---------------------------------

def create_app(
    model_loader: Callable = load_mlflow_model,
    version_loader: Callable = None
):

    if version_loader is None:
        def version_loader():
            client = mlflow.MlflowClient(
                tracking_uri=MLFLOW_TRACKING_URI
            )

            champion = client.get_model_version_by_alias(
                name="iris-classifier",
                alias="champion"
            )

            return champion.version

    # ---------------------------------
    # LIFESPAN
    # ---------------------------------

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        try:
            app.state.model = model_loader()
            app.state.model_version = version_loader()

            print(
                f"Loaded iris-classifier version "
                f"{app.state.model_version}"
            )

        except Exception as ex:
            raise RuntimeError(
                f"Failed to load champion model: {ex}"
            ) from ex

        yield

    # ---------------------------------
    # FASTAPI APPLICATION
    # ---------------------------------

    app = FastAPI(
        title="Iris Classification API",
        description="API for classifying iris flowers",
        version="3.0.0",
        lifespan=lifespan
    )

    # ---------------------------------
    # ROUTES
    # ---------------------------------

    @app.get("/")
    def home():
        return {
            "message": "API is ready to serve..."
        }

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy"
        }

    @app.post("/predict")
    def predict(data: IrisRequest, request: Request):

        model = request.app.state.model

        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = int(model.predict(features)[0])

        class_names = [
            "setosa",
            "versicolor",
            "virginica"
        ]

        if prediction not in range(len(class_names)):
            raise HTTPException(
                status_code=500,
                detail="Model returned an invalid prediction"
            )

        return {
            "prediction": prediction,
            "class_name": class_names[prediction],
            "model_version": request.app.state.model_version
        }

    return app


# ---------------------------------
# GLOBAL APPLICATION INSTANCE
# ---------------------------------

app = create_app()
