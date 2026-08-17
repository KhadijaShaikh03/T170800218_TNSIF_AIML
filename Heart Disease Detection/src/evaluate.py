import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from preprocessing import load_data, preprocess_data


def calculate_specificity(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    specificity = tn / (tn + fp)

    return specificity


def evaluate_models():

    print("Loading dataset...")

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = preprocess_data(df)

    models = joblib.load(
        "models/all_models.pkl"
    )

    results = []

    os.makedirs(
        "reports/confusion_matrices",
        exist_ok=True
    )

    best_model_name = None
    best_f1 = -1

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        y_pred = model.predict(X_test_scaled)

        y_probability = model.predict_proba(
            X_test_scaled
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        specificity = calculate_specificity(
            y_test,
            y_pred
        )

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Specificity": round(specificity, 4)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1-Score",
        ascending=False
    )

    os.makedirs("reports", exist_ok=True)

    results_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    print(
        results_df.to_string(index=False)
    )

    print("\nBest Model:")
    print(best_model_name)

    # Save best model
    best_model = models[best_model_name]

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print("\nBest model saved to:")
    print("models/best_model.pkl")


if __name__ == "__main__":
    evaluate_models()