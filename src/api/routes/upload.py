from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import os

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload training data (CSV or Excel)."""
    try:
        file_extension = file.filename.split(".")[-1]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Save file temporarily
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Read file to check validity
        if file_extension == "csv":
            df = pd.read_csv(file_path)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(file_path)
        else:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Unsupported file format. Upload CSV or Excel.")
        
        return {"filename": file.filename, "columns": df.columns.tolist(), "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
