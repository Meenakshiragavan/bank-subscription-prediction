# Bank Marketing Subscription Prediction

## Project Overview
A machine learning project that predicts whether a bank customer is likely to subscribe to a term deposit based on customer and marketing campaign information.

The project includes data preprocessing, model training, evaluation, decision-threshold selection, and a REST API built with FastAPI.

## Technologies
- Python
- Pandas, NumPy, Scikit-learn
- FastAPI and Uvicorn
- Joblib
- Swagger UI

## Machine Learning Workflow
1. Clean and explore the bank marketing dataset.
2. Split data into training, validation, and test sets.
3. Preprocess numerical and categorical features using a Scikit-learn pipeline.
4. Train a Random Forest classifier with class balancing.
5. Select a decision threshold using validation data to target at least 70% recall.
6. Evaluate the model on the held-out test set.
7. Serve predictions through a REST API.

The `duration` feature was excluded to avoid relying on information that may not be available before a marketing call ends.

## Model Evaluation

| Metric | Test Result |
|---|---:|
| ROC-AUC | 0.7735 |
| Recall | 64.9% |
| Precision | Approximately 27% |

The decision threshold was selected using validation data, not the test set. The lower precision reflects the trade-off involved in identifying more potential subscribers.

## Project Structure

```text
product_purchase_prediction/
├── data/
│   └── bank.csv
├── models/
│   ├── bank_subscription_model.joblib
│   └── decision_threshold.joblib
├── src/
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── requirements.txt
└── README.md
```

## Setup and Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

From the project root:

```bash
python -m uvicorn src.api:app --reload
```

### 4. Test the API

Open:

`http://127.0.0.1:8000/docs`

Expand `POST /predict`, select **Try it out**, enter customer details, and click **Execute**.

## Example Response

```json
{
  "prediction": "yes",
  "subscription_probability": 0.295,
  "decision_threshold": 0.245
}
```

The prediction is based on whether the estimated probability meets the selected decision threshold. This example is illustrative of an actual API response observed during testing.

## Key Learnings
- Building end-to-end Scikit-learn preprocessing and modeling pipelines
- Handling imbalanced classification data
- Understanding precision, recall, ROC-AUC, and decision thresholds
- Saving and loading trained models
- Creating and testing REST APIs with FastAPI
- Validating API inputs and handling unseen categorical values

## Future Improvements
- Add automated tests for the API and preprocessing pipeline.
- Containerize the application with Docker.
- Deploy the API to a cloud platform.
