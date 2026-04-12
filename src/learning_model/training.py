import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(base_dir, "data")
models_dir = os.path.join(base_dir, "models")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

dataset_path = os.path.join(data_dir, "floorball_dataset_ml.csv")
model_output_path = os.path.join(models_dir, "floorball_model.pkl")

TEST_SIZE_RATIO = 0.2

df = pd.read_csv(dataset_path)

print("ML dataset shape:", df.shape)
print(df.head())

target_column = "target_home_win"

X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()

categorical_features = ["competition_id"] if "competition_id" in X.columns else []
numeric_features = [col for col in X.columns if col not in categorical_features]

print("\nFeature columns:")
print(list(X.columns))
print("\nNumeric features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)


split_index = int(len(df) * (1 - TEST_SIZE_RATIO))

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()
y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print(f"\nTrain rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

logreg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42
    ))
])

models = {
    "LogisticRegression": logreg_pipeline,
    "RandomForest": rf_pipeline
}

results = []

for model_name, pipeline in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    results.append({
        "model_name": model_name,
        "accuracy": acc,
        "roc_auc": auc,
        "pipeline": pipeline
    })

best_result = max(results, key=lambda x: x["roc_auc"])
best_model = best_result["pipeline"]

joblib.dump(best_model, model_output_path)

print("\nBest model:", best_result["model_name"])
print(f"Best ROC AUC: {best_result['roc_auc']:.4f}")