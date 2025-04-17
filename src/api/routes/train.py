from fastapi import APIRouter, HTTPException
from src.components.model_training import train_lstm_for_dam

router = APIRouter()

@router.post("/train/{dam_name}")
def train_model(dam_name: str):
    """
    API to train an LSTM model for a specific dam.
    """
    try:
        response = train_lstm_for_dam(dam_name)
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
