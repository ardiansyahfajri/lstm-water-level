import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.components.model_inference import predict_next_5_days

UPLOAD_DIR = "data/uploads/"
router = APIRouter()

@router.post("/predict_uploaded/{dam_name}")
async def predict_from_uploaded_file(dam_name: str, file: UploadFile = File(...)):
    """
    Upload a 7-day CSV file and return the next 5-day inflow prediction.
    """
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        result = predict_next_5_days(dam_name, file_path)
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
