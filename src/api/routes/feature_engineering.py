from fastapi import APIRouter, HTTPException
from src.components.feature_engineering import apply_feature_engineering

router = APIRouter()

@router.get("/feature_engineering/{dam_name}")
def feature_engineering(dam_name: str):
    """
    API to apply feature engineering for a specific dam.
    """
    try:
        columns = apply_feature_engineering(dam_name)
        return {"message": f"Feature engineering completed for {dam_name}", "columns": columns}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying feature engineering: {str(e)}")
