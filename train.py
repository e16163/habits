"""
Quell — train.py  (updated for features.py)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from features import feature_names, N_FEATURES

print("Loading landmarks.csv …")
df = pd.read_csv("landmarks.csv").dropna()

print(f"  Total samples : {len(df)}")
print(f"  Touching (1)  : {(df['label'] == 1).sum()}")
print(f"  Not touching  : {(df['label'] == 0).sum()}")
print()

session_col = "session" if "session" in df.columns else None
feat_cols   = feature_names()

# Keep only columns that exist (backwards compat with old CSVs)
feat_cols = [c for c in feat_cols if c in df.columns]
print(f"Using {len(feat_cols)} features  (expected {N_FEATURES})")
if len(feat_cols) < N_FEATURES:
    print("  ⚠️  Some features missing — re-collect data with the new collect.py")
print()

X = df[feat_cols].values
y = df["label"].values

# ── Split ─────────────────────────────────────────────────────────────────────

if session_col:
    groups     = df[session_col].values
    n_sessions = len(np.unique(groups))
    if n_sessions > 1:
        print(f"Session-aware split across {n_sessions} sessions …")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        print("⚠️  Only 1 session — using random split for now.")
        print("   Run collect.py again (each run = new session) for a more honest split.\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
else:
    print("⚠️  No session column — using random split (less reliable).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

print(f"Train: {len(X_train)}   Test: {len(X_test)}\n")

# ── Train ─────────────────────────────────────────────────────────────────────

models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced", random_state=42))
    ]),
}

results = {}
for name, pipeline in models.items():
    print(f"Training {name} …")
    pipeline.fit(X_train, y_train)
    train_acc = (pipeline.predict(X_train) == y_train).mean()
    y_pred    = pipeline.predict(X_test)
    test_acc  = (y_pred == y_test).mean()
    gap       = train_acc - test_acc
    results[name] = dict(pipeline=pipeline, accuracy=test_acc,
                         train_acc=train_acc, y_pred=y_pred)
    print(f"  Train: {train_acc:.1%}   Test: {test_acc:.1%}   Gap: {gap:.1%}")
    if gap > 0.10:
        print("  ⚠️  Large gap — collect more varied data.")
    print(classification_report(y_test, y_pred,
          target_names=["not touching", "touching hair"]))

# ── Best model ────────────────────────────────────────────────────────────────

best_name     = max(results, key=lambda k: results[k]["accuracy"])
best_pipeline = results[best_name]["pipeline"]
print(f"\nBest: {best_name}  "
      f"(test {results[best_name]['accuracy']:.1%} / "
      f"train {results[best_name]['train_acc']:.1%})\n")

cm = confusion_matrix(y_test, results[best_name]["y_pred"])
print("Confusion matrix:")
print(f"               Predicted: no   Predicted: yes")
print(f"  Actual: no       {cm[0,0]:<8}     {cm[0,1]}")
print(f"  Actual: yes      {cm[1,0]:<8}     {cm[1,1]}\n")

# ── Save ──────────────────────────────────────────────────────────────────────

with open("quell_model.pkl", "wb") as f:
    pickle.dump({"pipeline": best_pipeline}, f)
print("Saved quell_model.pkl")
