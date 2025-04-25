import os
import string
from fastapi import APIRouter

MODELS_DIR = "models"

router = APIRouter()

@router.get("/models")
def list_models():
    if not os.path.exists(MODELS_DIR):
        return []
    models = [f.replace(".keras", "") for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
    return models