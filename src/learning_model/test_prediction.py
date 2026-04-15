from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "floorball_dataset_ml_with_roster.csv"
MODEL_PATH = BASE_DIR / "models" / "floorball_model_2.pkl"

TEST_SIZE_RATIO = 0.2
TARGET_COLUMN = "target_home_win"


def main():
    df = pd.read_csv(DATASET_PATH)
    model = joblib.load(MODEL_PATH)

    print("Loaded dataset:", DATASET_PATH)
    print("Loaded model:", MODEL_PATH)
    print("Dataset shape:", df.shape)

    df = df.dropna(subset=["home_roster_strength", "away_roster_strength"]).copy()
    print("Shape after dropping missing roster rows:", df.shape)

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

    split_index = int(len(df) * (1 - TEST_SIZE_RATIO))

    X_test = X.iloc[split_index:].copy()
    y_test = y.iloc[split_index:].copy()
    df_test = df.iloc[split_index:].copy()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n===== SAVED MODEL EVALUATION =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    df_test = df_test.copy()
    df_test["pred_home_win_prob"] = y_prob
    df_test["pred_class"] = y_pred
    df_test["correct"] = (df_test[TARGET_COLUMN] == df_test["pred_class"]).astype(int)
    df_test["confidence"] = (df_test["pred_home_win_prob"] - 0.5).abs()

    print("\nTop 15 most confident predictions:")
    cols_to_show = [
        "competition_id",
        "league",
        "pred_home_win_prob",
        "pred_class",
        "target_home_win",
        "correct",
        "confidence",
        "home_roster_strength",
        "away_roster_strength",
        "roster_strength_diff",
    ]
    print(
        df_test.sort_values("confidence", ascending=False)[cols_to_show]
        .head(15)
        .to_string(index=False)
    )

    high_conf = df_test[df_test["confidence"] >= 0.20].copy()
    very_high_conf = df_test[df_test["confidence"] >= 0.30].copy()

    if len(high_conf) > 0:
        print(f"\nHigh confidence picks (|p-0.5| >= 0.20): {len(high_conf)}")
        print(f"Accuracy: {high_conf['correct'].mean():.4f}")

    if len(very_high_conf) > 0:
        print(f"\nVery high confidence picks (|p-0.5| >= 0.30): {len(very_high_conf)}")
        print(f"Accuracy: {very_high_conf['correct'].mean():.4f}")


if __name__ == "__main__":
    main()