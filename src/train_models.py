"""Train classification baselines and save reproducible evaluation artifacts."""
import json
import warnings
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                            roc_auc_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.data import load_data, FEATURES, TARGET, ARTIFACTS

def make_models():
    models = {
        "Logistic Regression": Pipeline([("prep", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])),
                                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),
        "Decision Tree": Pipeline([("prep", SimpleImputer(strategy="median")),
                                   ("model", DecisionTreeClassifier(max_depth=7, min_samples_leaf=5, class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("prep", SimpleImputer(strategy="median")),
                                   ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42))]),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([("prep", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
                objective="binary:logistic", eval_metric="logloss", n_jobs=-1,
                random_state=42, verbosity=0))])
    except ImportError:
        warnings.warn("xgboost is not installed; skipping XGBoost. Install requirements.txt to enable it.")
    return models

def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    results = {"dataset": {"rows": len(df), "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
                           "features": FEATURES, "split": "stratified 80/20", "random_state": 42}, "models": {}}
    for name, pipeline in make_models().items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        try:
            score = float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
        except (AttributeError, ValueError):
            score = None
        key = name.lower().replace(" ", "_")
        joblib.dump(pipeline, ARTIFACTS / f"{key}.joblib")
        results["models"][name] = {
            "accuracy": float(accuracy_score(y_test, pred)), "f1": float(f1_score(y_test, pred, zero_division=0)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)), "roc_auc": score,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
        print(f"{name:22} accuracy={results['models'][name]['accuracy']:.3f}  F1={results['models'][name]['f1']:.3f}")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved metrics and models to {ARTIFACTS}")

if __name__ == "__main__":
    main()
