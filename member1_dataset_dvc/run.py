
import pandas as pd
import joblib  # type: ignore[import-untyped]
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Load dataset
wine = load_wine(as_frame=True)
df = wine.frame.copy()

df["target_name"] = df["target"].map(
    dict(enumerate(wine.target_names))
)

# Step 2: Save dataset
Path("data").mkdir(exist_ok=True)
df.to_csv("data/wine.csv", index=False)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Step 3: Perform EDA
print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

print("\nClass distribution:")
print(df["target_name"].value_counts())

# Step 4: Separate features and target
X = df.drop(columns=["target", "target_name"])
y = df["target"]

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 6: Define three models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4, random_state=42
    ),

    "Gaussian Naive Bayes": GaussianNB()
}

# Step 7: Train and compare
results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted"
    )
    recall = recall_score(
        y_test, predictions, average="weighted"
    )
    f1 = f1_score(
        y_test, predictions, average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

# Step 8: Display comparison
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="F1 Score", ascending=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

# Step 9: Select best model
best_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

Path("models").mkdir(exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

joblib.dump(best_model, "models/best_model.joblib")
comparison.to_csv("metrics/comparison.csv", index=False)

print("\nBest Model:", best_name)
print("Model saved successfully!")
print("Comparison saved successfully!")
