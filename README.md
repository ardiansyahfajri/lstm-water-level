# 🌊 LSTM Water Level Prediction

This project is an end-to-end system for predicting water levels using an **LSTM model**. It consists of:

- **Backend**: Built with **FastAPI** for APIs (data ingestion, processing, training, predictions).
- **Frontend**: Developed with **Streamlit** for a user-friendly interface.

---

## 🚀 Features

### 🔧 Backend (FastAPI)

- **API Endpoints** for data ingestion, processing, training, predictions, and model management.
- Supports uploading datasets, training LSTM models, and making predictions via API.

### 🖥️ Frontend (Streamlit)

- Upload data, trigger training, and view predictions through an intuitive interface.
- Interactive charts and visualizations.

---

## 🛠️ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/lstm-water-level.git
cd lstm-water-level
```

### 2️⃣ Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
uvicorn src.api.app:app --reload
```

FastAPI server will run at: [http://localhost:8000](http://localhost:8000)

### 3️⃣ Frontend Setup

In a new terminal, run:

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit app will run at: [http://localhost:8501](http://localhost:8501)

### 4️⃣ Docker Setup (Optional)

You can run both frontend and backend using Docker Compose.

```bash
docker-compose up --build
```

## Example Workflow

1. Upload historical water level data via Streamlit.
2. Train an LSTM model on the data.
3. Use the model to predict future water levels.
4. Visualize predictions and trends.
