
import pandas as pd
import joblib

# 1. Load the saved pipeline and threshold
saved_model = joblib.load("models/bank_subscription_model.joblib")

model_pipeline = saved_model["pipeline"]
threshold = saved_model["threshold"]

# 2. Create one example customer
customer = pd.DataFrame([{
    "age": 35,
    "job": "technician",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 1200,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}])

# 3. Find the probability for the "yes" class
yes_index = list(model_pipeline.classes_).index("yes")
probability = model_pipeline.predict_proba(customer)[0, yes_index]

# 4. Apply the saved threshold
prediction = "yes" if probability >= threshold else "no"

print(f"Subscription probability: {probability:.2%}")
print(f"Prediction: {prediction}")
print(f"Decision threshold: {threshold}")
