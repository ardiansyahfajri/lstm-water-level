from fastapi import FastAPI
from src.api.routes.upload import router as upload_router
from src.api.routes.process import router as process_router
from src.api.routes.feature_engineering import router as feature_router
from src.api.routes.data_ingestion import router as ingestion_router
from src.api.routes.train import router as train_router
from src.api.routes.predict import router as predict_router


app = FastAPI()

app.include_router(upload_router, prefix="/api")
app.include_router(process_router, prefix="/api")
app.include_router(ingestion_router, prefix="/api")
app.include_router(feature_router, prefix="/api")
app.include_router(train_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
