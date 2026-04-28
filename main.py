import os
import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 1. Initialize FastAPI
app = FastAPI(title="Credit Card Fraud Detection API (XGBoost)")

# 2. Define the Input Schema (Matches Kaggle Dataset Structure)
class TransactionData(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

# 3. Global Variables for Model and Scalers
model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    try:
        # Load XGBoost model (using the modern .json format if available)
        # If you saved with joblib, use joblib.load instead
        model = joblib.load("final_xgboost_model.pkl")
        scaler = joblib.load("amount_time_scaler.pkl")
        
        print("✅ SUCCESS: XGBoost model and Scaler loaded.")
    except Exception as e:
        print(f"❌ ERROR: Loading failed. Details: {e}")

@app.get("/")
def home():
    return {"status": "Online", "model": "XGBoost + SMOTE Pipeline"}

@app.post("/predict")
def predict_fraud(data: TransactionData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        
        # Preprocessing: Scale 'Time' and 'Amount' 
        X_test_amount_time = scaler.transform(input_df[['Amount','Time']])

        X_test = input_df.copy()
        X_test['Amount_Scaled']  = X_test_amount_time[:,0]
        X_test['Time_Scaled']    = X_test_amount_time[:,1]
        X_test = X_test.drop(['Amount','Time'], axis=1)
        
        # Prediction
        # XGBoost predict_proba returns [prob_normal, prob_fraud]
        prediction_prob = model.predict_proba(X_test)[0][1]
        
        # Decision Threshold (Adjust based on your CV results!)
        threshold = 0.91
        is_fraud = bool(prediction_prob > threshold)

        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(float(prediction_prob), 4),
            "threshold_used": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
