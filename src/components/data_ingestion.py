import os
import pandas as pd

DATA_DIR = "data/uploads/"
PROCESSED_DIR = "data/processed/"

def load_uploaded_file(dam_name: str):
    """
    Dynamically finds the correct file format (.csv or .xlsx) for a dam.
    """
    possible_extensions = [".csv", ".xlsx"]
    file_path = None

    # Search for the correct file
    for ext in possible_extensions:
        potential_path = os.path.join(DATA_DIR, f"{dam_name}{ext}")
        if os.path.exists(potential_path):
            file_path = potential_path
            break

    if file_path is None:
        raise FileNotFoundError(f"No uploaded file found for dam: {dam_name} in {DATA_DIR}")

    # Load the correct file format
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format for {dam_name}")

    return df

def save_processed_data(dam_name: str, df: pd.DataFrame):
    """
    Saves the processed dataset for a specific dam.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, f"{dam_name}.csv")
    df.to_csv(processed_path, index=False)
    return processed_path

def process_data_ingestion(dam_name: str):
    """
    Loads, cleans, and saves the processed dataset for a specific dam.
    """
    df = load_uploaded_file(dam_name)

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    save_processed_data(dam_name, df)
    return df.columns.tolist()
