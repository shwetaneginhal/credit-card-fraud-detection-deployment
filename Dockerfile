# Step 1: Use Python 3.11 for modern library support
FROM python:3.11-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Install system dependencies for XGBoost
# libgomp1 is required for XGBoost's parallel processing
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your model, scaler, and code
# Ensure your local files are named exactly like this
COPY final_xgboost_model.pkl .
COPY amount_time_scaler.pkl .
COPY main.py .

# Step 6: Expose port and run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]