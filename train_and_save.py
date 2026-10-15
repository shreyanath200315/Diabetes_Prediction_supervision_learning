"""
Builds the `diabetes_deployment/` folder (and saved_models/all_tuning_results.csv)
from diabetes.csv, using the corrected pipeline:

  zeros -> NaN -> split -> median imputer (train only) -> Glucose/BMI ratio -> scaler
  -> tune 9 models with 5-fold CV (F1) -> save the best by CV F1

Run from the folder that contains diabetes.csv:   python train_and_save.py
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("xgboost not installed -> skipping XGBoost (pip install xgboost to include it)")

DATA_FILE = "diabetes.csv"
INVALID = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

if not os.path.exists(DATA_FILE):
    sys.exit(f"Cannot find {DATA_FILE}. Put it in this folder and run again.")

df = pd.read_csv(DATA_FILE)

# --- data check: was the file already imputed? -------------------------------
zeros = int(sum((df[c] == 0).sum() for c in INVALID))
if zeros == 0:
    print("\n*** WARNING: no zeros in Glucose/BP/Skin/Insulin/BMI. This CSV looks already imputed")
    print("*** (often with class-wise medians = target leakage). Scores will be inflated.")
    print("*** Use the ORIGINAL Pima dataset (e.g. Kaggle 'Pima Indians Diabetes Database').\n")

df[INVALID] = df[INVALID].replace(0, np.nan)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
base_features = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# --- preprocessing (fit on TRAIN only) ---------------------------------------
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=base_features, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=base_features, index=X_test.index)

for d in (X_train, X_test):
    d["Glucose_BMI_Ratio"] = d["Glucose"] / d["BMI"]

selected_features = list(X_train.columns)
scaler = StandardScaler()
Xtr = pd.DataFrame(scaler.fit_transform(X_train[selected_features]), columns=selected_features)
Xte = pd.DataFrame(scaler.transform(X_test[selected_features]), columns=selected_features)

# --- models + grids ------------------------------------------------------------
grids = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42),
                            {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]}),
    "KNN": (KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "p": [1, 2]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42),
                      {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 10],
                       "min_samples_leaf": [1, 3, 5], "class_weight": [None, "balanced"]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"n_estimators": [100, 200], "max_depth": [5, 10, None],
                       "min_samples_split": [2, 10], "min_samples_leaf": [1, 3],
                       "class_weight": [None, "balanced"]}),
    "SVM": (SVC(probability=True, random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "class_weight": [None, "balanced"]}),
    "Gaussian Naive Bayes": (GaussianNB(), {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1],
                           "max_depth": [1, 2, 3], "min_samples_leaf": [3, 5]}),
    "AdaBoost": (AdaBoostClassifier(random_state=42),
                 {"n_estimators": [50, 100, 150], "learning_rate": [0.1, 0.5, 1.0]}),
}
if XGBClassifier is not None:
    grids["XGBoost"] = (XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1),
                        {"n_estimators": [50, 100, 150], "max_depth": [2, 3, 5],
                         "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0]})

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows, tuned = [], {}

for name, (model, grid) in grids.items():
    print(f"Tuning {name} ...", flush=True)
    search = GridSearchCV(model, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
    search.fit(Xtr, y_train)
    best = search.best_estimator_
    tuned[name] = best
    pred = best.predict(Xte)
    prob = best.predict_proba(Xte)[:, 1]
    rows.append({
        "Model": name,
        "Best CV F1": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, pred),
        "Test Precision": precision_score(y_test, pred, zero_division=0),
        "Test Recall": recall_score(y_test, pred, zero_division=0),
        "Test F1": f1_score(y_test, pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, prob),
        "Best Parameters": str(search.best_params_),
    })
    print(f"   CV F1 = {search.best_score_:.4f} | test acc = {rows[-1]['Test Accuracy']:.4f}")

results = pd.DataFrame(rows).sort_values("Best CV F1", ascending=False)
print("\n", results.drop(columns=["Best Parameters"]).round(4).to_string(index=False))

best_name = results.iloc[0]["Model"]          # chosen by CV, NOT by the test set
print("\nSelected model (best CV F1):", best_name)

# --- save ------------------------------------------------------------------------
os.makedirs("diabetes_deployment", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
joblib.dump(tuned[best_name], "diabetes_deployment/diabetes_model.pkl")
joblib.dump(scaler, "diabetes_deployment/scaler.pkl")
joblib.dump(imputer, "diabetes_deployment/imputer.pkl")
joblib.dump(selected_features, "diabetes_deployment/selected_features.pkl")
joblib.dump(base_features, "diabetes_deployment/base_features.pkl")
results.to_csv("saved_models/all_tuning_results.csv", index=False)
print("\nSaved: diabetes_deployment/ and saved_models/all_tuning_results.csv")
print("Now run:  streamlit run app.py")
