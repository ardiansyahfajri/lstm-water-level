import os
from fastapi import APIRouter, HTTPException

MODELS_DIR = "models"

router = APIRouter()

@router.get("/models")
def list_models():
    if not os.path.exists(MODELS_DIR):
        return []
    models = [f.replace(".keras", "") for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
    return models

@router.delete("/models/{model_name}")
def delete_model(model_name: str):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        os.remove(model_path)
        return {'message': f'Model {model_name} deleted successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")
        