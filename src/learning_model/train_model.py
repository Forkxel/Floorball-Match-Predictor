from pathlib import Path
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
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "floorball_dataset_ml_with_roster.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "floorball_model_2.pkl"

TEST_SIZE_RATIO = 0.2
TARGET_COLUMN = "target_home_win"

REQUIRED_ROSTER_COLUMNS = [
    "home_roster_strength",
    "away_roster_strength",
]


def main():
    if not DATASET_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print("Original dataset shape:", df.shape)

    missing_required = [col for col in REQUIRED_ROSTER_COLUMNS if col not in df.columns]
    if missing_required:
        raise RuntimeError(f"Missing roster columns: {missing_required}")

    df = df.dropna(subset=REQUIRED_ROSTER_COLUMNS).copy()

    print("Dataset shape after dropping missing roster rows:", df.shape)
    print(df.head(5).to_string())

    if TARGET_COLUMN not in df.columns:
        raise RuntimeError(f"Missing target column: {TARGET_COLUMN}")

    drop_columns = []

    for col in [
        "season_norm",
        "roster_prev_season",
        "home_roster_missing",
        "away_roster_missing",
    ]:
        if col in df.columns:
            drop_columns.append(col)

    extra_drop_columns = [
        "season",
        "season_norm",
        "roster_prev_season",
        "home_roster_missing",
        "away_roster_missing",
    ]

    drop_columns = [col for col in extra_drop_columns if col in df.columns]

    X = df.drop(columns=[TARGET_COLUMN] + drop_columns).copy()
    y = df[TARGET_COLUMN].copy()

    preferred_categorical = ["competition_id", "league"]
    categorical_features = [col for col in preferred_categorical if col in X.columns]

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

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("Train/test split failed — dataset too small.")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logreg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )),
    ])

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])

    models = {
        "LogisticRegression": logreg_pipeline,
        "RandomForest": rf_pipeline,
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
            "pipeline": pipeline,
        })

    best_result = max(results, key=lambda x: x["roc_auc"])
    best_model = best_result["pipeline"]

    joblib.dump(best_model, MODEL_OUTPUT_PATH)

    print("\nBest model:", best_result["model_name"])
    print(f"Best ROC AUC: {best_result['roc_auc']:.4f}")
    print(f"Saved model: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()