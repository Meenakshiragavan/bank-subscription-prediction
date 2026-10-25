from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Bank Subscription Prediction API",
    description="Predicts whether a bank customer is likely to subscribe.",
    version="1.0.0"
)


# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the saved model and decision threshold
MODEL_PATH = BASE_DIR / "models" / "bank_subscription_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "decision_threshold.joblib"

# Load the saved files into memory
model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

print("ML model loaded successfully!")
print("Decision threshold:", threshold)

# Define the customer data expected by our API
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


# Create the prediction endpoint
@app.post("/predict")
def predict_subscription(customer: CustomerData):
    # Convert the validated customer data into a DataFrame
    customer_df = pd.DataFrame([customer.model_dump()])

    # Get the model's predicted probability for "yes"
    probability = model.predict_proba(customer_df)[0][
        list(model.classes_).index("yes")
    ]

    # Apply our recall-focused decision threshold
    prediction = "yes" if probability >= threshold else "no"

    return {
        "prediction": prediction,
        "subscription_probability": round(float(probability), 4),
        "decision_threshold": threshold
    }
