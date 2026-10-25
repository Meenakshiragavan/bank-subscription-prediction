
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


# ---------------------------------
# 1. Load the dataset
# ---------------------------------

data = pd.read_csv("data/bank.csv", sep=";")

print("Dataset shape:", data.shape)
print(data.head())


# ---------------------------------
# 2. Separate features and target
# ---------------------------------

X = data.drop(columns=["y", "duration"])
y = data["y"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ---------------------------------
# 3. Split into train, validation and test
# ---------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=40,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=40,
    stratify=y_temp
)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))


# ---------------------------------
# 4. Identify feature types
# ---------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)


# ---------------------------------
# 5. Build preprocessing pipelines
# ---------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ---------------------------------
# 6. Build and train the balanced model
# ---------------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=40,
        class_weight="balanced"
    ))
])

model_pipeline.fit(X_train, y_train)

print("\nBalanced Random Forest trained successfully!")


# ---------------------------------
# Logistic Regression experiment
# ---------------------------------

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=40
    ))
])

logistic_pipeline.fit(X_train, y_train)

print("\n=== LOGISTIC REGRESSION: VALIDATION RESULTS ===")

logistic_predictions = logistic_pipeline.predict(X_val)
print(classification_report(y_val, logistic_predictions))

logistic_probabilities = logistic_pipeline.predict_proba(X_val)[:, 1]

logistic_roc_auc = roc_auc_score(
    (y_val == "yes").astype(int),
    logistic_probabilities
)

print("Logistic Regression ROC-AUC:", round(logistic_roc_auc, 4))

threshold = 0.4

logistic_threshold_predictions = [
    "yes" if probability >= threshold else "no"
    for probability in logistic_probabilities
]

print("\nLogistic Regression at threshold 0.4")
print("Precision:", round(
    precision_score(
        y_val, logistic_threshold_predictions, pos_label="yes"
    ), 3
))
print("Recall:", round(
    recall_score(
        y_val, logistic_threshold_predictions, pos_label="yes"
    ), 3
))
print("F1-score:", round(
    f1_score(
        y_val, logistic_threshold_predictions, pos_label="yes"
    ), 3
))



#---------------------------------------------------------
# Compare Logistic Regression at the same recall target
y_val_binary = (y_val == "yes").astype(int)

logistic_precision, logistic_recall, logistic_thresholds = (
    precision_recall_curve(
        y_val_binary,
        logistic_probabilities
    )
)
target_recall = 0.70
valid_logistic = np.where(
    logistic_recall[:-1] >= target_recall
)[0]

if len(valid_logistic) == 0:
    print("Logistic Regression cannot reach the target recall.")
else:
    best_logistic_index = valid_logistic[
        np.argmax(logistic_precision[:-1][valid_logistic])
    ]

    print("\nLOGISTIC REGRESSION: RECALL-FOCUSED")
    print(
        "Threshold:",
        round(logistic_thresholds[best_logistic_index], 4)
    )
    print(
        "Precision:",
        round(logistic_precision[best_logistic_index], 3)
    )
    print(
        "Recall:",
        round(logistic_recall[best_logistic_index], 3)
    )


# ---------------------------------
# 7. Evaluate on validation data
# ---------------------------------

y_val_pred = model_pipeline.predict(X_val)

print("\nBALANCED MODEL: DEFAULT THRESHOLD")
print(classification_report(y_val, y_val_pred))

y_val_prob = model_pipeline.predict_proba(X_val)[:, 1]

val_roc_auc = roc_auc_score(
    (y_val == "yes").astype(int),
    y_val_prob
)

print("Validation ROC-AUC:", round(val_roc_auc, 4))


# ---------------------------------
# 8. Select threshold for high recall
# ---------------------------------

import numpy as np
from sklearn.metrics import precision_recall_curve

y_val_binary = (y_val == "yes").astype(int)

precision_values, recall_values, thresholds = precision_recall_curve(
    y_val_binary,
    y_val_prob
)

target_recall = 0.70

# Each threshold corresponds to precision_values[i] and recall_values[i]
valid_indices = np.where(recall_values[:-1] >= target_recall)[0]

if len(valid_indices) == 0:
    print(f"No threshold achieves recall >= {target_recall:.0%}")
    selected_threshold = None
else:
    # Among thresholds meeting the recall target, choose highest precision
    best_index = valid_indices[
        np.argmax(precision_values[:-1][valid_indices])
    ]

    selected_threshold = thresholds[best_index]

    print("\nRECALL-FOCUSED THRESHOLD")
    print("Selected threshold:", round(selected_threshold, 4))
    print("Validation precision:", round(precision_values[best_index], 3))
    print("Validation recall:", round(recall_values[best_index], 3))

# ---------------------------------
# 9. Confusion matrix at threshold 0.4
# ---------------------------------


y_val_pred_final = np.where(
    y_val_prob >= selected_threshold,
    "yes",
    "no"
)

print("\nSELECTED THRESHOLD:", selected_threshold)
print("Validation confusion matrix:")
print(confusion_matrix(
    y_val,
    y_val_pred_final,
    labels=["no", "yes"]
))

print(classification_report(y_val, y_val_pred_final))

# ---------------------------------
# 10. Compare against the baseline
# ---------------------------------

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_val)

print("\nBASELINE RESULTS")
print(classification_report(
    y_val,
    baseline_predictions,
    zero_division=0
))



# ---------------------------------
# 11. Final test evaluation
# ---------------------------------

import joblib
from pathlib import Path

# Use the threshold selected using validation data
final_threshold = selected_threshold

# Predict probabilities on the untouched test set
y_test_prob = model_pipeline.predict_proba(X_test)[:, 1]

# Convert probabilities into subscriber predictions
y_test_pred = [
    "yes" if prob >= final_threshold else "no"
    for prob in y_test_prob
]

print("\n=== FINAL TEST RESULTS ===")
print("Decision threshold:", round(final_threshold, 4))

print(classification_report(
    y_test,
    y_test_pred,
    labels=["no", "yes"],
    zero_division=0
))

print("Test ROC-AUC:", round(
    roc_auc_score(
        (y_test == "yes").astype(int),
        y_test_prob
    ),
    4
))

print("Test confusion matrix:")
print(confusion_matrix(
    y_test,
    y_test_pred,
    labels=["no", "yes"]
))

# ---------------------------------
# 12. Save model and threshold
# ---------------------------------

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model_pipeline,
    "models/bank_subscription_model.joblib"
)

joblib.dump(
    final_threshold,
    "models/decision_threshold.joblib"
)

print("\nModel and threshold saved successfully!")
