import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")
# API_BASE = 'http://localhost:8000/api'

st.set_page_config(page_title='Dam Water Level Prediction', layout="centered")
st.title('Dam Water Level Forecasting')

# Model Selection
st.header('Manage Models')

model_action = st.radio("Choose:", ["Select Existing Model", "Create New Model"])
model_name = None

if model_action == "Select Existing Model":
    response = requests.get(f"{API_BASE}/models")
    if response.status_code == 200:
        model_list = response.json()
        if model_list:
            model_name = st.selectbox("Available models:", model_list)
            if st.button("Delete Selected Model"):
                delete_response = requests.delete(f"{API_BASE}/models/{model_name}")
                if delete_response.status_code == 200:
                    st.success(f"Model '{model_name}' deleted successfully!")
                    st.rerun()  # Refresh model list
                else:
                    st.error(f"Failed to delete model: {delete_response.text}")
        else:
            st.info("No models found.")
    else:
        st.error("Failed to fetch model list.")
        
else:
    model_name = st.text_input('Enter a new model name:')
    training_file = st.file_uploader('Upload training data (.csv or .xlsx)', type=['csv', 'xlsx'])
    
    if st.button('Train Model') and model_name and training_file:
        try:    
            files = {'file': training_file}
            
            steps = [
                ("Uploading...", f"{API_BASE}/upload/", "POST"),
                ("Processing...", f"{API_BASE}/process/{model_name}", "GET"),
                ("Ingesting...", f"{API_BASE}/ingest/{model_name}", "GET"),
                ("Feature Engineering...", f"{API_BASE}/feature_engineering/{model_name}", "GET"),
                ("Training...", f"{API_BASE}/train/{model_name}", "POST")
            ]
            
            for label, endpoint, method in steps:
                with st.spinner(label):
                    if method == "POST":
                        resp = requests.post(endpoint, files=files)
                    else:
                        resp = requests.get(endpoint)
                    
                    if resp.status_code != 200:
                        st.error(f'{label} failed: {resp.text}')
                        break
            else:
                st.success(f'Model {model_name} created and trained successfully!')
    
        except Exception as e:
            st.error(f'{str(e)}')
            
            
# Prediction
if model_name and model_action == 'Select Existing Model':
    st.header('Make Predictions')
    pred_file = st.file_uploader('Upload data for prediction input (.csv or .xlsx)', type=['csv', 'xlsx'])
    if st.button('Predict') and pred_file:
        try:
            files = {'file': pred_file}
            pred_resp = requests.post(f'{API_BASE}/predict_uploaded/{model_name}', files=files)
            if pred_resp.status_code == 200:
                forecast = pred_resp.json()['forecast']
                forecast_df = pd.DataFrame.from_dict(forecast, orient='index')
                st.success('Prediction Completed!')
                st.subheader(f'Predicted Water Levels for {model_name.title()}')
                st.dataframe(forecast_df)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_df.index.tolist() + forecast_df.index[::-1].tolist(),
                    y=forecast_df["upper"].tolist() + forecast_df["lower"][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(0, 123, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval 95%',
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df["mean"],
                    mode="lines+markers",
                    marker=dict(size=8, color="blue", symbol="circle"),
                    name="Mean TMA",
                    line=dict(color="blue"),
                    hovertemplate="%{x}<br>• Mean TMA: %{y:.2f} m<extra></extra>"
                ))
                y_min = forecast_df["lower"].min()
                y_max = forecast_df["upper"].max()
                padding = (y_max - y_min) * 0.1
                fig.update_layout(
                    title='5-Day TMA Forecast with 95% Confidence Interval',
                    xaxis_title="Date",
                    yaxis_title="TMA (m)",
                    yaxis=dict(range=[y_min - padding, y_max + padding]),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f'Prediction failed: {pred_resp.text}')
        except Exception as e:
            st.error(f'{str(e)}')
        