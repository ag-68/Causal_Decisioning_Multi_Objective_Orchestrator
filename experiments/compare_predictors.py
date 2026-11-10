"""Choose predictor settings on the validation slice and score the held-out slice.

The live dashboard fits on train plus validation. This script holds the
validation people out while it compares settings, then refits the chosen
settings on train plus validation and scores the untouched test people.
Squared error is the shared score. Accuracy is recorded beside it because a
coin that is usually 0 makes accuracy look high.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from app.data import (
    ACTIONS,
    N_ACTIONS,
    SEGMENT_NAMES,
    SEGMENT_TO_CODE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    covariate_matrix,
    feature_matrix,
    load_hillstrom,
    make_cross_section,
    prepare_demo,
)
from app.models import make_hgb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "predictor_fit.json"

TREE_GRID = [
    {"max_iter": 40, "max_depth": 3, "min_samples_leaf": 50, "l2_regularization": 1.0, "learning_rate": 0.08},
    {"max_iter": 40, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.01, "learning_rate": 0.08},
    {"max_iter": 60, "max_depth": 8, "min_samples_leaf": 20, "l2_regularization": 0.01, "learning_rate": 0.08},
    {"max_iter": 80, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.01, "learning_rate": 0.08},
    {"max_iter": 80, "max_depth": 4, "min_samples_leaf": 40, "l2_regularization": 0.1, "learning_rate": 0.05},
    {"max_iter": 120, "max_depth": 3, "min_samples_leaf": 40, "l2_regularization": 0.1, "learning_rate": 0.05},
]
SVM_GRID = [
    {"kernel": "linear", "C": 0.1, "epsilon": 0.0},
    {"kernel": "linear", "C": 1.0, "epsilon": 0.0},
    {"kernel": "linear", "C": 1.0, "epsilon": 0.1},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.05, "n_components": 64},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.2, "n_components": 64},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.1, "n_components": 128},
]
MLP_GRID = [
    {"hidden": [64], "alpha": 1e-4, "lr": 1e-3, "max_iter": 40},
    {"hidden": [128], "alpha": 1e-4, "lr": 1e-3, "max_iter": 40},
    {"hidden": [128], "alpha": 1e-3, "lr": 1e-3, "max_iter": 40},
    {"hidden": [64], "alpha": 1e-4, "lr": 1e-2, "max_iter": 25},
]
DEEP_GRID = [
    {"hidden": [128, 64], "alpha": 1e-4, "lr": 1e-3, "max_iter": 40},
    {"hidden": [128, 64, 32], "alpha": 1e-4, "lr": 1e-3, "max_iter": 40},
    {"hidden": [256, 128, 64], "alpha": 1e-4, "lr": 1e-3, "max_iter": 30},
]


class _Mean:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value, dtype=np.float64)


class _Scaled:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, features: np.ndarray) -> np.ndarray:
        coded = features if self.scaler is None else self.scaler.transform(features)
        return np.clip(self.model.predict(coded), 0.0, 1.0)


def _build(kind: str, params: dict, seed: int):
    if kind == "trees":
        return make_hgb(seed=seed, **params), None
    scaler = StandardScaler()
    if kind == "svm" and params["kernel"] == "linear":
        model = LinearSVR(
            C=params["C"],
            epsilon=params["epsilon"],
            loss="squared_epsilon_insensitive",
            dual=False,
            max_iter=4000,
            random_state=seed,
        )
        return model, scaler
    if kind == "svm":
        model = Pipeline(
            [
                (
                    "map",
                    Nystroem(
                        kernel="rbf",
                        gamma=params["gamma"],
                        n_components=params["n_components"],
                        random_state=seed,
                    ),
                ),
                (
                    "svr",
                    LinearSVR(
                        C=params["C"],
                        loss="squared_epsilon_insensitive",
                        dual=False,
                        max_iter=4000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        return model, scaler
    model = MLPRegressor(
        hidden_layer_sizes=tuple(params["hidden"]),
        activation="relu",
        alpha=params["alpha"],
        learning_rate_init=params["lr"],
        max_iter=params["max_iter"],
        batch_size=256,
        early_stopping=False,
        random_state=seed,
        tol=1e-4,
    )
    return model, scaler


def _fit_one(kind: str, params: dict, features: np.ndarray, outcome: np.ndarray, seed: int):
    if len(features) < 15:
        value = float(outcome.mean()) if len(outcome) else 0.0
        return _Mean(value)
    model, scaler = _build(kind, params, seed)
    coded = features if scaler is None else scaler.fit_transform(features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(coded, outcome)
    return _Scaled(model, scaler)


def _sums(outcome: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(outcome, dtype=np.float64)
    n = int(len(y))
    if n == 0:
        return {"n": 0, "sse": 0.0, "hits": 0, "positives": 0, "log_loss": 0.0}
    clipped = np.clip(pred, 1e-6, 1.0 - 1e-6)
    log_loss = float(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).sum())
    return {
        "n": n,
        "sse": float(np.square(y - pred).sum()),
        "hits": int(np.sum((pred >= 0.5) == (y >= 0.5))),
        "positives": int(np.sum(y >= 0.5)),
        "log_loss": log_loss,
    }


def _blank():
    return {"n": 0, "sse": 0.0, "hits": 0, "positives": 0, "log_loss": 0.0}


def _add(total: dict, part: dict) -> None:
    for key in total:
        total[key] += part[key]


def _rates(total: dict) -> dict:
    n = max(int(total["n"]), 1)
    majority = max(total["positives"], n - total["positives"]) / n
    return {
        "mse": total["sse"] / n,
        "accuracy": total["hits"] / n,
        "log_loss": total["log_loss"] / n,
        "positive_rate": total["positives"] / n,
        "majority_accuracy": majority,
        "n": int(total["n"]),
    }


def _packs(features: np.ndarray, actions: np.ndarray, outcome: np.ndarray, split: np.ndarray, names: tuple[str, ...], target: str):
    packs = []
    for arm, name in enumerate(names):
        mask = actions == arm
        train = mask & (split == SPLIT_TRAIN)
        val = mask & (split == SPLIT_VAL)
        test = mask & (split == SPLIT_TEST)
        fit = mask & (split != SPLIT_TEST)
        packs.append(
            {
                "arm": name,
                "target": target,
                "x_train": features[train],
                "y_train": outcome[train],
                "x_val": features[val],
                "y_val": outcome[val],
                "x_test": features[test],
                "y_test": outcome[test],
                "x_fit": features[fit],
                "y_fit": outcome[fit],
            }
        )
    return packs


def _score_split(models: list, packs: list, x_key: str, y_key: str) -> dict:
    overall = _blank()
    by_target: dict[str, dict] = {}
    for model, pack in zip(models, packs):
        part = _sums(pack[y_key], model.predict(pack[x_key]))
        _add(overall, part)
        bucket = by_target.setdefault(pack["target"], _blank())
        _add(bucket, part)
    report = _rates(overall)
    report["by_target"] = {name: _rates(bucket) for name, bucket in by_target.items()}
    return report


def _search(kind: str, grid: list[dict], packs: list, seed: int) -> list[dict]:
    rows = []
    for params in grid:
        started = time.perf_counter()
        models = [
            _fit_one(kind, params, pack["x_train"], pack["y_train"], seed + index)
            for index, pack in enumerate(packs)
        ]
        train = _score_split(models, packs, "x_train", "y_train")
        val = _score_split(models, packs, "x_val", "y_val")
        row = {
            "params": params,
            "train_mse": train["mse"],
            "val_mse": val["mse"],
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(
            f"  {kind} {params} train {train['mse']:.6f} val {val['mse']:.6f} ({row['seconds']:.1f}s)",
            flush=True,
        )
    return rows


def _refit(kind: str, params: dict, packs: list, latency_x: np.ndarray, seed: int) -> dict:
    started = time.perf_counter()
    models = [
        _fit_one(kind, params, pack["x_fit"], pack["y_fit"], seed + index) for index, pack in enumerate(packs)
    ]
    fit_seconds = time.perf_counter() - started
    train = _score_split(models, packs, "x_fit", "y_fit")
    test = _score_split(models, packs, "x_test", "y_test")
    batch = latency_x[: min(1024, len(latency_x))]
    for model in models:
        model.predict(batch)
    timed = time.perf_counter()
    repeats = 5
    for _ in range(repeats):
        for model in models:
            model.predict(batch)
    predict_ms = (time.perf_counter() - timed) / repeats * 1000.0
    return {
        "fit_on": "train_plus_validation",
        "train": train,
        "test": test,
        "fit_seconds": fit_seconds,
        "predict_ms_1024": predict_ms,
        "batch_size": int(len(batch)),
    }


def _run_method(kind: str, grid: list[dict], packs: list, latency_x: np.ndarray, seed: int) -> dict:
    print(kind, flush=True)
    search = _search(kind, grid, packs, seed)
    chosen = min(search, key=lambda row: row["val_mse"])
    refit = _refit(kind, chosen["params"], packs, latency_x, seed)
    print(
        f"  chosen val {chosen['val_mse']:.6f} refit train {refit['train']['mse']:.6f} "
        f"test {refit['test']['mse']:.6f} predict {refit['predict_ms_1024']:.2f} ms",
        flush=True,
    )
    return {"chosen": chosen["params"], "search": search, "refit": refit}


def _baseline(packs: list, latency_x: np.ndarray) -> dict:
    models = [_Mean(float(pack["y_fit"].mean()) if len(pack["y_fit"]) else 0.0) for pack in packs]
    batch = latency_x[: min(1024, len(latency_x))]
    timed = time.perf_counter()
    for _ in range(5):
        for model in models:
            model.predict(batch)
    return {
        "chosen": {"rule": "training positive rate of that message"},
        "search": [],
        "refit": {
            "fit_on": "train_plus_validation",
            "train": _score_split(models, packs, "x_fit", "y_fit"),
            "test": _score_split(models, packs, "x_test", "y_test"),
            "fit_seconds": 0.0,
            "predict_ms_1024": (time.perf_counter() - timed) / 5 * 1000.0,
            "batch_size": int(len(batch)),
        },
    }


def _outcome_counts(outcome: np.ndarray, split: np.ndarray) -> dict:
    counts = {}
    for name, code in (("train", SPLIT_TRAIN), ("validation", SPLIT_VAL), ("test", SPLIT_TEST)):
        mask = split == code
        counts[name] = {"rows": int(mask.sum()), "positives": int(outcome[mask].sum())}
    return counts


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    seed = 0
    n = 2000 if args.quick else 0
    grids = {
        "trees": TREE_GRID[:1] if args.quick else TREE_GRID,
        "svm": SVM_GRID[:1] if args.quick else SVM_GRID,
        "mlp": MLP_GRID[:1] if args.quick else MLP_GRID,
        "deep_mlp": DEEP_GRID[:1] if args.quick else DEEP_GRID,
    }
    full, source = load_hillstrom(n_fallback=max(n, 1), seed=seed, fallback=False)
    demo = prepare_demo(full, n=n, seed=seed)
    cross = make_cross_section(demo, seed=seed)
    split = cross["split"].to_numpy(int)
    print(f"source {source} rows {len(cross)}", flush=True)
    sim_x = feature_matrix(cross, cross["f"].to_numpy(np.float64))
    actions = cross["action"].to_numpy(int)
    sim_packs = _packs(sim_x, actions, cross["y_conversion"].to_numpy(np.float64), split, ACTIONS, "buy")
    sim_packs += _packs(sim_x, actions, cross["y_optout"].to_numpy(np.float64), split, ACTIONS, "stop")
    email_codes = cross["segment"].map(SEGMENT_TO_CODE).to_numpy(int)
    email_x = covariate_matrix(cross)
    email_packs = _packs(
        email_x,
        email_codes,
        cross["conversion"].to_numpy(np.float64),
        split,
        SEGMENT_NAMES,
        "conversion",
    )
    test_x = sim_x[split == SPLIT_TEST]
    payload = {
        "source": source,
        "rows": int(len(cross)),
        "quick": bool(args.quick),
        "protocol": (
            "Settings are chosen by the lowest validation squared error. "
            "That search fits on the training people only. The chosen settings "
            "are then refit on training plus validation, and the test numbers "
            "below are from that refit. The test people are not used to choose."
        ),
        "accuracy_note": (
            "Accuracy is the share of coins on the same side of 0.5 as the prediction. "
            "When almost every coin is 0, a predictor that always says 0 scores the share of zeros."
        ),
        "counts": {
            "simulator_buy": _outcome_counts(cross["y_conversion"].to_numpy(np.float64), split),
            "simulator_stop": _outcome_counts(cross["y_optout"].to_numpy(np.float64), split),
            "hillstrom_buy": _outcome_counts(cross["conversion"].to_numpy(np.float64), split),
        },
        "simulator": {},
        "hillstrom": {},
    }
    print("counts", json.dumps(payload["counts"]), flush=True)
    for kind, grid in grids.items():
        payload["simulator"][kind] = _run_method(kind, grid, sim_packs, test_x, seed)
        _dump(args.out, payload)
    payload["simulator"]["training_average"] = _baseline(sim_packs, test_x)
    _dump(args.out, payload)
    email_test_x = email_x[split == SPLIT_TEST]
    for kind, grid in grids.items():
        payload["hillstrom"][kind] = _run_method(kind, grid, email_packs, email_test_x, seed + 100)
        _dump(args.out, payload)
    payload["hillstrom"]["training_average"] = _baseline(email_packs, email_test_x)
    _dump(args.out, payload)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
