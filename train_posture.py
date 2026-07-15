"""Train, validate, calibrate, and profile a personalized posture model."""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from posture_features import (
    FEATURE_SCHEMA_VERSION,
    N_FEATURES,
    feature_names,
    posture_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="posture_landmarks.csv")
    parser.add_argument("--output", default="posture_model.pkl")
    parser.add_argument(
        "--view",
        choices=("front", "side", "all"),
        default="front",
        help="train one consistent view unless you have several sessions of each",
    )
    return parser.parse_args()


def extra_trees(seed: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def random_forest(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def robust_logistic(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scale", RobustScaler(quantile_range=(10, 90))),
            (
                "clf",
                LogisticRegression(
                    C=0.25,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def model_candidates() -> dict[str, object]:
    return {
        "Extra Trees": extra_trees(),
        "Random Forest": random_forest(),
        "Robust Logistic": robust_logistic(),
        "Soft Voting Ensemble": VotingClassifier(
            estimators=[
                ("extra", extra_trees(43)),
                ("forest", random_forest(43)),
                ("linear", robust_logistic(43)),
            ],
            voting="soft",
            weights=(2, 2, 1),
            # The tree members already parallelize internally; fitting voting
            # members sequentially avoids nested process pools on macOS.
            n_jobs=1,
        ),
    }


def make_splits(X, y, groups):
    session_count = len(np.unique(groups))
    if session_count >= 3:
        folds = min(5, session_count)
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=42
        )
        splits = list(splitter.split(X, y, groups))
        mode = f"{folds}-fold stratified session cross-validation"
    elif session_count == 2:
        splitter = GroupShuffleSplit(n_splits=2, test_size=0.5, random_state=42)
        splits = list(splitter.split(X, y, groups))
        mode = "two held-out-session evaluations"
    else:
        folds = min(5, int(np.bincount(y, minlength=2).min()))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        splits = list(splitter.split(X, y))
        mode = f"{folds}-fold frame validation (collect more sessions)"

    valid = [
        (train_index, test_index)
        for train_index, test_index in splits
        if len(np.unique(y[train_index])) == 2 and len(np.unique(y[test_index])) == 2
    ]
    if not valid:
        raise ValueError(
            "Validation could not place both labels in train and test folds. "
            "Record good and bad examples in every collection session."
        )
    return valid, mode


def probability_of_bad(model, X) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


def tune_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.linspace(0.30, 0.70, 81)
    scored = []
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        balanced = balanced_accuracy_score(y_true, prediction)
        bad_f1 = f1_score(y_true, prediction, pos_label=1, zero_division=0)
        # Prefer a threshold near 0.5 when several perform identically.
        scored.append((balanced, bad_f1, -abs(threshold - 0.5), threshold))
    return float(max(scored)[-1])


def cross_validated_probabilities(model, X, y, splits) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.full(len(y), np.nan, dtype=np.float64)
    visits = np.zeros(len(y), dtype=np.int32)
    accumulated = np.zeros(len(y), dtype=np.float64)
    for train_index, test_index in splits:
        fitted = clone(model).fit(X[train_index], y[train_index])
        accumulated[test_index] += probability_of_bad(fitted, X[test_index])
        visits[test_index] += 1
    covered = visits > 0
    probabilities[covered] = accumulated[covered] / visits[covered]
    return probabilities, covered


def robust_profile(X: np.ndarray, y: np.ndarray) -> dict:
    center = np.median(X, axis=0)
    mad = np.median(np.abs(X - center), axis=0) * 1.4826
    # Protect nearly constant features without washing out naturally small
    # normalized-coordinate measurements.
    floor = np.maximum(np.std(X, axis=0) * 0.05, 1e-4)
    scale = np.maximum(mad, floor)
    deviation = np.median(np.abs((X - center) / scale), axis=1)

    good_center = np.median(X[y == 0], axis=0)
    bad_center = np.median(X[y == 1], axis=0)
    return {
        "feature_center": center.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "ood_limit": float(max(np.quantile(deviation, 0.99) * 1.35, 3.5)),
        "good_profile": posture_metrics(good_center),
        "bad_profile": posture_metrics(bad_center),
    }


def tree_importances(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    if isinstance(model, VotingClassifier):
        importances = [
            np.asarray(estimator.feature_importances_)
            for estimator in model.estimators_
            if hasattr(estimator, "feature_importances_")
        ]
        if importances:
            return np.mean(importances, axis=0)
    return None


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input).dropna()
    if args.view != "all" and "view" in frame:
        frame = frame[frame["view"].astype(str) == args.view].copy()
        if frame.empty:
            raise ValueError(f"No {args.view!r}-view samples found in {args.input}")

    names = feature_names()
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} current posture features. "
            "Re-collect it with collect_posture.py."
        )
    if len(names) != N_FEATURES:
        raise RuntimeError("Posture feature schema is internally inconsistent")

    X = frame[names].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must include both good (0) and bad (1) examples")
    class_counts = np.bincount(y, minlength=2)
    if min(class_counts) < 40:
        raise ValueError("Collect at least 40 diverse frames for each class before training")

    groups = (
        frame["session"].astype(str).to_numpy()
        if "session" in frame
        else np.array(["one_session"] * len(frame))
    )
    splits, validation_mode = make_splits(X, y, groups)
    unique_ratio = len(np.unique(np.round(X, 4), axis=0)) / len(X)

    print(f"Samples: {len(frame)}  good: {class_counts[0]}  bad: {class_counts[1]}")
    print(f"Features: {len(names)}  sessions: {len(np.unique(groups))}")
    print(f"Validation: {validation_mode}")
    print(f"Pose diversity: {unique_ratio:.1%} unique frames\n")
    if unique_ratio < 0.35:
        print("Warning: many frames are near-duplicates; collect more pose variation.\n")

    results = {}
    for name, candidate in model_candidates().items():
        probabilities, covered = cross_validated_probabilities(
            candidate, X, y, splits
        )
        y_eval = y[covered]
        p_eval = probabilities[covered]
        threshold = tune_threshold(y_eval, p_eval)
        prediction = (p_eval >= threshold).astype(int)
        results[name] = {
            "template": candidate,
            "probability": p_eval,
            "actual": y_eval,
            "prediction": prediction,
            "threshold": threshold,
            "balanced_accuracy": balanced_accuracy_score(y_eval, prediction),
            "accuracy": accuracy_score(y_eval, prediction),
            "bad_f1": f1_score(y_eval, prediction, pos_label=1),
        }
        score = results[name]
        print(
            f"{name:22s} balanced {score['balanced_accuracy']:.1%}  "
            f"accuracy {score['accuracy']:.1%}  bad-F1 {score['bad_f1']:.1%}  "
            f"threshold {threshold:.2f}"
        )

    best_name = max(
        results,
        key=lambda name: (
            results[name]["balanced_accuracy"],
            results[name]["bad_f1"],
            results[name]["accuracy"],
            name == "Soft Voting Ensemble",
        ),
    )
    best = results[best_name]
    print(f"\nSelected: {best_name}")
    print(
        classification_report(
            best["actual"], best["prediction"], target_names=["good", "bad"]
        )
    )
    print("Cross-validated confusion matrix (rows actual, columns predicted):")
    print(confusion_matrix(best["actual"], best["prediction"]))

    # Validation estimates generalization; the shipped model should then learn
    # from every labeled session, not discard the validation folds.
    final_model = clone(best["template"]).fit(X, y)
    importance = tree_importances(final_model)
    if importance is not None:
        order = np.argsort(importance)[::-1][:12]
        print("\nMost useful features:")
        for index in order:
            print(f"  {names[index]:36s} {importance[index]:.3f}")

    profile = robust_profile(X, y)
    views = sorted(frame["view"].astype(str).unique()) if "view" in frame else [args.view]
    bundle = {
        "pipeline": final_model,
        "feature_names": names,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_version": 2,
        "label_names": {0: "good", 1: "bad"},
        "views": views,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "decision_threshold": best["threshold"],
        "profile": profile,
        "evaluation": {
            "model": best_name,
            "validation": validation_mode,
            "folds": len(splits),
            "balanced_accuracy": best["balanced_accuracy"],
            "accuracy": best["accuracy"],
            "bad_f1": best["bad_f1"],
            "unique_frame_ratio": unique_ratio,
        },
    }
    with open(args.output, "wb") as file:
        pickle.dump(bundle, file)
    print(f"\nRefit {best_name} on all {len(X)} samples")
    print(f"Saved advanced model bundle to {args.output}")


if __name__ == "__main__":
    main()
