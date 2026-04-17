# Credit Card Fraud Detection API

This repository contains a production-ready REST API designed to detect fraudulent credit card transactions in real-time. The system uses an **XGBoost** classifier trained on the ULB Credit Card Fraud dataset, served via **FastAPI**, and fully containerized with **Docker**.

## 🚀 Features
- **High Performance:** Utilizes XGBoost, SMOTE and Cross Validation techniques for the highly imbalanced nature of financial fraud data.
- **Pre-processing Pipeline:** Integrated `RobustScaler` for handling outliers in `Time` and `Amount` features.
- **Automated Documentation:** Interactive API docs via Swagger UI (`/docs`).
- **Containerized Deployment:** Docker-ready for consistent behavior across development and production environments.
- **Balanced Evaluation:** Used two alternative thresholds, the **F1-Score** and **Cost-optimal method** to ensure a balance between catching fraud (Recall) and minimizing false alarms (Precision).

---

## Decision Threshold & Final Logic

Both optimization (F1 score and Cost optimal) methods suggest an unusually high decision threshold (above 0.90). This indicates that the XGBoost model has high discriminative power, clearly separating fraud from legitimate transactions with high confidence. 
While a threshold of 0.94 maximizes the **F1-Score**, the 0.91 threshold was chosen for **Cost-Optimization**. By providing a more inclusive safety margin, this setting minimizes the total projected financial loss to $1,944.98, prioritizing the detection of high-cost fraudulent transactions over pure statistical balance.

## 📁 Project Structure
```text
.
├── main.py                    # FastAPI application & API logic
├── fraud_detection_run.ipynb  # EDA, SMOTE analysis, Modelling, Precision-Recall curve, Threshold optimization 
├── final_xgboost_model.pkl    # Trained XGBoost model 
├── amount_time_scaler.pkl     # Saved RobustScaler for feature scaling
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration for deployment
└── README.md                  # Project documentation

```
## 🛠️ Getting Started

### 1. Prerequisites
* **Python 3.10+** (if running locally)
* **Docker** (if running as a container)

### 2. Local Setup
If you want to run the API directly on your host machine:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
### 3.Docker deployment
To build and run the containerized version:
```bash
# Build the Docker image
docker build -t fraud-detector-xgb .

# Run the container (maps port 8000 on host to 8000 in container)
docker run -d -p 8000:8000 --name fraud_api_container fraud-detector-xgb

#open in browser
http://localhost:8000/docs

#to stop docker:
sudo docker stop fraud_api_container

#to restart:
sudo docker start fraud_api_container

#to remove docker:
sudo docker rm fraud_api_container
```
## 📖 Usage

Once the API is running, access the interactive documentation at:
[http://localhost:8000/docs](http://localhost:8000/docs)

### API Endpoint: `POST /predict`
Submit a JSON payload containing 30 features (`Time`, `V1-V28`, and `Amount`).

**Sample Request:**

```json
{
  "Time": 406.0,
  "V1": -2.312226, "V2": 1.951992, "V3": -1.609850, "V4": 3.997905, "V5": -0.522187,
  "V6": -1.426545, "V7": -2.537387, "V8": 1.391657, "V9": -2.770089, "V10": -2.772272,
  "V11": 3.202033, "V12": -2.899907, "V13": -0.595221, "V14": -4.289253, "V15": 0.389724,
  "V16": -1.140747, "V17": -2.830055, "V18": -0.016822, "V19": 0.416955, "V20": 0.126910,
  "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198, "V25": 0.044519,
  "V26": 0.177839, "V27": 0.261145, "V28": -0.143276,
  "Amount": 1.00
}
```
## 🛑 Troubleshooting

* **Port Conflict:** If port 8000 is taken, stop existing containers using `docker rm -f fraud_api_container`.
* **Docker Logs:** To debug startup issues, use `docker logs fraud_api_container`.
* **Feature Names:** Ensure the input JSON includes all 30 features (`Time`, `V1`–`V28`, `Amount`) in that exact order.

