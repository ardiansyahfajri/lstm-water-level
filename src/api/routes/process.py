import os
import pandas as pd
from fastapi import APIRouter, HTTPException

DATA_DIR = "data/uploads/"
PROCESSED_DIR = "data/processed/"

router = APIRouter()

def load_uploaded_file(filename: str):
    """
    Tries to find the correct file format (.csv or .xlsx) and load it.
    """
    possible_extensions = [".csv", ".xlsx"]
    file_path = None

    # Search for the correct file
    for ext in possible_extensions:
        potential_path = os.path.join(DATA_DIR, f"{filename}{ext}")
        if os.path.exists(potential_path):
            file_path = potential_path
            break

    if file_path is None:
        raise FileNotFoundError(f"No uploaded file found for: {filename}")

    # Load file
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format for {filename}")

    return df

@router.get("/process/{filename}")
def process_file(filename: str):
    """
    API to process an uploaded file and save it as processed data.
    """
    try:
        df = load_uploaded_file(filename)

        # Clean Data (Remove Empty Rows, Standardize Columns)
        df.dropna(inplace=True)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        # Save Processed Data
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        processed_path = os.path.join(PROCESSED_DIR, f"{filename}.csv")
        df.to_csv(processed_path, index=False)

        return {"message": f"Processing complete for {filename}", "columns": df.columns.tolist()}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
