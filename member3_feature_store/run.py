
# MEMBER 3: FEATURE ENGINEERING AND FEATURE STORE
# Dataset: Wine Classification

import pandas as pd
import joblib

from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# STEP 1: LOAD DATASET

wine = load_wine(as_frame=True)

df = wine.frame.copy()

print("Wine dataset loaded successfully!")

print("Dataset shape:", df.shape)

# STEP 2: SAVE ORIGINAL DATASET

Path("data").mkdir(exist_ok=True)

df.to_csv("data/wine.csv", index=False)

print("\nOriginal dataset saved successfully!")

# STEP 3: SEPARATE FEATURES AND TARGET

X = df.drop(columns="target").copy()

y = df["target"]

print("\nOriginal number of features:", X.shape[1])

# STEP 4: FEATURE ENGINEERING

# Feature 1: Alcohol to total phenols ratio

X["alcohol_phenol_ratio"] = (
    X["alcohol"] / (X["total_phenols"] + 1e-8)
)

# Feature 2: Flavanoids ratio

X["flavanoids_ratio"] = (
    X["flavanoids"] /
    (X["nonflavanoid_phenols"] + 1e-8)
)

# Feature 3: Color intensity to proline ratio

X["color_proline_ratio"] = (
    X["color_intensity"] / (X["proline"] + 1e-8)
)

# Feature 4: Square of total phenols

X["total_phenols_squared"] = (
    X["total_phenols"] ** 2
)

print("\nFeature engineering completed!")

print("New number of features:", X.shape[1])

# STEP 5: CREATE FEATURE STORE

Path("feature_store").mkdir(exist_ok=True)

feature_data = X.copy()

feature_data["target"] = y.values

feature_data.to_csv(
    "feature_store/wine_features.csv",
    index=False
)

print("\nFeatures stored successfully!")

# STEP 6: RETRIEVE FEATURES FROM FEATURE STORE

stored_data = pd.read_csv(
    "feature_store/wine_features.csv"
)

X = stored_data.drop(columns="target")

y = stored_data["target"]

print("Features retrieved successfully!")

print("Retrieved feature shape:", X.shape)

# STEP 7: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining and testing data created!")

# STEP 8: DEFINE MACHINE LEARNING MODELS

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC())
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])

}

# STEP 9: TRAIN AND EVALUATE MODELS

results = []

trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1

    })

    trained_models[name] = model

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))

# STEP 10: MODEL COMPARISON

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")

print(comparison.to_string(index=False))

# STEP 11: SELECT BEST MODEL

best_name = comparison.iloc[0]["Model"]

best_model = trained_models[best_name]

print("\nBest Model:", best_name)

# STEP 12: SAVE BEST MODEL

Path("models").mkdir(exist_ok=True)

Path("metrics").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "feature_columns": list(X.columns)
    },
    "models/best_model.joblib"
)

# STEP 13: SAVE MODEL COMPARISON

comparison.to_csv(
    "metrics/comparison.csv",
    index=False
)

print("\nBest model saved successfully!")

print("Comparison report saved successfully!")

print("\nFEATURE STORE PRACTICAL COMPLETED!")
