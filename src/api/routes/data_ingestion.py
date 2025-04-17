from fastapi import APIRouter, HTTPException
from src.components.data_ingestion import process_data_ingestion

router = APIRouter()

@router.get("/ingest/{dam_name}")
def ingest_data(dam_name: str):
    """
    API to process and save ingested data for a specific dam.
    """
    try:
        columns = process_data_ingestion(dam_name)
        return {"message": f"Data ingestion completed for {dam_name}", "columns": columns}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
