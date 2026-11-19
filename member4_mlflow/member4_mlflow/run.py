
# MEMBER 4: WINE CLASSIFICATION USING MLFLOW

import os
import warnings
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")

# STEP 1: CREATE REQUIRED DIRECTORIES

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Directories created successfully!")

# STEP 2: LOAD WINE DATASET

wine = load_wine(as_frame=True)

X = wine.data
y = wine.target

# Save dataset as CSV
dataset = wine.frame.copy()
dataset.to_csv("data/wine.csv", index=False)

print("\nWine dataset loaded successfully!")
print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Number of classes:", len(wine.target_names))

# STEP 3: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# STEP 4: DEFINE CLASSIFICATION MODELS

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf"))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])
}

print("\nFive classification models defined successfully!")

# STEP 5: CONFIGURE MLFLOW

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("Wine_Classification_Experiment")

results = []

# STEP 6: TRAIN AND EVALUATE MODELS

print("\nStarting model training...")

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # Log parameters in MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        # Log metrics in MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log trained model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        })

        print("\nModel:", model_name)
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

# STEP 7: COMPARE MODELS

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# STEP 8: SELECT BEST MODEL

best_model_name = results_df.iloc[0]["Model"]

best_model = models[best_model_name]

# Train selected model
best_model.fit(X_train, y_train)

# Save best model
joblib.dump(
    best_model,
    "models/best_model.joblib"
)

print("\n========== BEST MODEL ==========")
print("Best Model:", best_model_name)
print(
    "Best F1 Score:",
    round(results_df.iloc[0]["F1_Score"], 4)
)

print("\nDataset saved successfully!")
print("Model comparison saved successfully!")
print("Best model saved successfully!")
print("MLflow experiment tracking completed!")
