import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "heart_disease_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Save scaler
scaler_path = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)

print("Scaler saved successfully.")


# --------------------------------------------------
# DEFINE MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": CalibratedClassifierCV(
        SVC(random_state=42),
        ensemble=False
    )
}


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train_scaled,
        y_train
    )

    trained_models[name] = model

    print(f"{name} training completed.")


# --------------------------------------------------
# SAVE ALL MODELS
# --------------------------------------------------

all_models_path = os.path.join(
    MODEL_DIR,
    "all_models.pkl"
)

joblib.dump(
    trained_models,
    all_models_path
)

print("\nAll models saved successfully.")


# --------------------------------------------------
# SAVE FINAL BEST MODEL
# --------------------------------------------------

best_model = trained_models["Random Forest"]

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)

print("Best model (Random Forest) saved successfully.")

print("\nTraining pipeline completed.")