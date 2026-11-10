"""Train the same four families with log loss and score buyers and non-buyers apart.

Squared error was the previous score. This run is separate. Each model is
trained to minimize log loss on every row. A non-buyer with predicted chance
q costs -ln(1-q). A buyer costs -ln(q). Settings are chosen by the average
log loss on the validation people. The chosen settings are refit on training
plus validation and scored on the untouched test people.

The report splits that same log loss into the buyer rows and the non-buyer
rows, and it checks that the average predicted chance stays near the real
rate. The live dashboard is left as it is.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.data import (
    ACTIONS,
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
from app.models import contrasts
from app.ope import qini_curve

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "predictor_logloss.json"

TREE_GRID = [
    {"max_iter": 40, "max_depth": 3, "min_samples_leaf": 50, "l2_regularization": 1.0, "learning_rate": 0.08},
    {"max_iter": 40, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.01, "learning_rate": 0.08},
    {"max_iter": 60, "max_depth": 8, "min_samples_leaf": 20, "l2_regularization": 0.01, "learning_rate": 0.08},
    {"max_iter": 80, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.01, "learning_rate": 0.08},
    {"max_iter": 80, "max_depth": 4, "min_samples_leaf": 40, "l2_regularization": 0.1, "learning_rate": 0.05},
    {"max_iter": 120, "max_depth": 3, "min_samples_leaf": 40, "l2_regularization": 0.1, "learning_rate": 0.05},
]
LINEAR_GRID = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
KERNEL_GRID = [
    {"C": 1.0, "gamma": 0.05, "n_components": 64},
    {"C": 1.0, "gamma": 0.2, "n_components": 64},
    {"C": 10.0, "gamma": 0.1, "n_components": 128},
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


class _Chance:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, features: np.ndarray) -> np.ndarray:
        coded = features if self.scaler is None else self.scaler.transform(features)
        proba = self.model.predict_proba(coded)
        classes = [int(label) for label in self.model.classes_]
        if 1 not in classes:
            return np.zeros(len(features), dtype=np.float64)
        return np.clip(proba[:, classes.index(1)], 0.0, 1.0)


def _build(kind: str, params: dict, seed: int):
    if kind == "trees":
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=params["max_iter"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            early_stopping=False,
            random_state=seed,
        )
        return model, None
    scaler = StandardScaler()
    if kind == "linear":
        model = LogisticRegression(C=params["C"], max_iter=400, solver="lbfgs")
        return model, scaler
    if kind == "kernel":
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
                ("logit", LogisticRegression(C=params["C"], max_iter=400, solver="lbfgs")),
            ]
        )
        return model, scaler
    model = MLPClassifier(
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
    labels = np.asarray(outcome, dtype=int)
    if len(features) < 15 or len(np.unique(labels)) < 2:
        value = float(labels.mean()) if len(labels) else 0.0
        return _Mean(value)
    model, scaler = _build(kind, params, seed)
    coded = features if scaler is None else scaler.fit_transform(features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        model.fit(coded, labels)
    return _Chance(model, scaler)


def _blank():
    return {"n": 0, "log_sum": 0.0, "pos_n": 0, "pos_log_sum": 0.0, "neg_n": 0, "neg_log_sum": 0.0, "sse": 0.0, "pred_sum": 0.0}


def _add(total: dict, part: dict) -> None:
    for key in total:
        total[key] += part[key]


def _parts(outcome: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(outcome, dtype=np.float64)
    chance = np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if len(y) == 0:
        return _blank()
    row = -(y * np.log(chance) + (1.0 - y) * np.log(1.0 - chance))
    bought = y >= 0.5
    return {
        "n": int(len(y)),
        "log_sum": float(row.sum()),
        "pos_n": int(bought.sum()),
        "pos_log_sum": float(row[bought].sum()) if bought.any() else 0.0,
        "neg_n": int((~bought).sum()),
        "neg_log_sum": float(row[~bought].sum()) if (~bought).any() else 0.0,
        "sse": float(np.square(y - chance).sum()),
        "pred_sum": float(chance.sum()),
    }


def _rates(total: dict) -> dict:
    n = max(int(total["n"]), 1)
    pos_n = int(total["pos_n"])
    neg_n = int(total["neg_n"])
    return {
        "n": int(total["n"]),
        "positives": pos_n,
        "log_loss": total["log_sum"] / n,
        "buyer_log_loss": (total["pos_log_sum"] / pos_n) if pos_n else None,
        "nonbuyer_log_loss": (total["neg_log_sum"] / neg_n) if neg_n else None,
        "mse": total["sse"] / n,
        "mean_prediction": total["pred_sum"] / n,
        "positive_rate": pos_n / n,
    }


def _score_split(models: list, packs: list, x_key: str, y_key: str) -> dict:
    overall = _blank()
    for model, pack in zip(models, packs):
        _add(overall, _parts(pack[y_key], model.predict(pack[x_key])))
    return _rates(overall)


def _packs(features, actions, outcome, split, names):
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


def _search(kind, grid, packs, seed):
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
            "train_log_loss": train["log_loss"],
            "val_log_loss": val["log_loss"],
            "val_buyer_log_loss": val["buyer_log_loss"],
            "val_nonbuyer_log_loss": val["nonbuyer_log_loss"],
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(
            f"  {kind} {params} train {train['log_loss']:.4f} val {val['log_loss']:.4f} "
            f"val_buyer {val['buyer_log_loss']} ({row['seconds']:.1f}s)",
            flush=True,
        )
    return rows


def _qini(models, x_test, actions, outcome, n_actions):
    columns = [model.predict(x_test) for model in models]
    if len(columns) != n_actions:
        return None
    tau = contrasts(np.column_stack(columns))
    if n_actions == 4:
        propensity = np.full(4, 0.25)
    else:
        counts = np.bincount(actions, minlength=n_actions).astype(np.float64)
        propensity = counts / max(float(counts.sum()), 1.0)
    curve = qini_curve(tau, actions, outcome, propensity)
    end = float(curve["uplift"][-1])
    uplift = np.asarray(curve["uplift"], dtype=np.float64)
    random_line = np.asarray(curve["random"], dtype=np.float64)
    interior = slice(1, -1)
    above = int(np.sum(uplift[interior] > random_line[interior] + 1e-6))
    compared = int(len(uplift[interior]))
    return {
        "auuc": curve["auuc"],
        "end": end,
        "above_straight_path": end / 2.0,
        "area_above_straight_path": curve["auuc"] - end / 2.0,
        "points_above_straight_path": above,
        "interior_points": compared,
    }


def _refit(kind, params, packs, x_test, actions, outcome, seed, with_qini):
    started = time.perf_counter()
    models = [_fit_one(kind, params, pack["x_fit"], pack["y_fit"], seed + index) for index, pack in enumerate(packs)]
    fit_seconds = time.perf_counter() - started
    train = _score_split(models, packs, "x_fit", "y_fit")
    test = _score_split(models, packs, "x_test", "y_test")
    batch = x_test[: min(1024, len(x_test))]
    for model in models:
        model.predict(batch)
    timed = time.perf_counter()
    for _ in range(5):
        for model in models:
            model.predict(batch)
    report = {
        "fit_on": "train_plus_validation",
        "train": train,
        "test": test,
        "fit_seconds": fit_seconds,
        "predict_ms_1024": (time.perf_counter() - timed) / 5.0 * 1000.0,
        "batch_size": int(len(batch)),
    }
    if with_qini:
        report["test_qini"] = _qini(models, x_test, actions, outcome, n_actions=len(packs))
    return report


def _run_method(kind, grid, packs, x_test, actions, outcome, seed, with_qini):
    print(kind, flush=True)
    search = _search(kind, grid, packs, seed)
    chosen = min(search, key=lambda row: row["val_log_loss"])
    refit = _refit(kind, chosen["params"], packs, x_test, actions, outcome, seed, with_qini)
    print(
        f"  chosen val {chosen['val_log_loss']:.4f} "
        f"train {refit['train']['log_loss']:.4f} test {refit['test']['log_loss']:.4f} "
        f"test_buyer {refit['test']['buyer_log_loss']}",
        flush=True,
    )
    return {"chosen": chosen["params"], "search": search, "refit": refit}


def _baseline(packs, x_test, actions, outcome, with_qini):
    models = [_Mean(float(pack["y_fit"].mean()) if len(pack["y_fit"]) else 0.0) for pack in packs]
    report = {
        "chosen": {"rule": "buy rate of that message on the fit rows"},
        "search": [],
        "refit": {
            "fit_on": "train_plus_validation",
            "train": _score_split(models, packs, "x_fit", "y_fit"),
            "test": _score_split(models, packs, "x_test", "y_test"),
            "fit_seconds": 0.0,
            "predict_ms_1024": 0.0,
            "batch_size": int(min(1024, len(x_test))),
        },
    }
    if with_qini:
        report["refit"]["test_qini"] = _qini(models, x_test, actions, outcome, n_actions=len(packs))
    return report


def _counts(outcome, split):
    counts = {}
    for name, code in (("train", SPLIT_TRAIN), ("validation", SPLIT_VAL), ("test", SPLIT_TEST)):
        mask = split == code
        counts[name] = {"rows": int(mask.sum()), "positives": int(outcome[mask].sum())}
    return counts


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _task(name, packs, x_test, actions, outcome, grids, seed, payload, path, with_qini):
    print(f"== {name}", flush=True)
    block = {}
    for kind, grid in grids.items():
        block[kind] = _run_method(kind, grid, packs, x_test, actions, outcome, seed, with_qini)
        payload[name] = block
        _dump(path, payload)
    block["message_average"] = _baseline(packs, x_test, actions, outcome, with_qini)
    payload[name] = block
    _dump(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    seed = 0
    n = 2000 if args.quick else 0
    grids = {
        "trees": TREE_GRID[:1] if args.quick else TREE_GRID,
        "linear": LINEAR_GRID[:1] if args.quick else LINEAR_GRID,
        "kernel": KERNEL_GRID[:1] if args.quick else KERNEL_GRID,
        "mlp": MLP_GRID[:1] if args.quick else MLP_GRID,
        "deep_mlp": DEEP_GRID[:1] if args.quick else DEEP_GRID,
    }
    full, source = load_hillstrom(n_fallback=max(n, 1), seed=seed, fallback=False)
    demo = prepare_demo(full, n=n, seed=seed)
    cross = make_cross_section(demo, seed=seed)
    split = cross["split"].to_numpy(int)
    sim_x = feature_matrix(cross, cross["f"].to_numpy(np.float64))
    actions = cross["action"].to_numpy(int)
    buy = cross["y_conversion"].to_numpy(np.float64)
    stop = cross["y_optout"].to_numpy(np.float64)
    email_codes = cross["segment"].map(SEGMENT_TO_CODE).to_numpy(int)
    email_x = covariate_matrix(cross)
    email_y = cross["conversion"].to_numpy(np.float64)
    test = split == SPLIT_TEST
    payload = {
        "source": source,
        "rows": int(len(cross)),
        "quick": bool(args.quick),
        "loss": "log loss on every row. A buyer costs -ln(q). A non-buyer costs -ln(1-q).",
        "protocol": (
            "Settings minimize validation log loss, fit on the training people only. "
            "The chosen settings are refit on training plus validation. "
            "Test numbers use the untouched 8,000. Buyer log loss averages -ln(q) "
            "over rows whose coin is 1. Non-buyer log loss averages -ln(1-q) over the rest."
        ),
        "counts": {
            "simulator_buy": _counts(buy, split),
            "simulator_stop": _counts(stop, split),
            "hillstrom_buy": _counts(email_y, split),
        },
    }
    print("counts", json.dumps(payload["counts"]), flush=True)
    _task(
        "simulator_buy",
        _packs(sim_x, actions, buy, split, ACTIONS),
        sim_x[test],
        actions[test],
        buy[test],
        grids,
        seed,
        payload,
        args.out,
        with_qini=True,
    )
    _task(
        "simulator_stop",
        _packs(sim_x, actions, stop, split, ACTIONS),
        sim_x[test],
        actions[test],
        stop[test],
        grids,
        seed + 50,
        payload,
        args.out,
        with_qini=False,
    )
    _task(
        "hillstrom_buy",
        _packs(email_x, email_codes, email_y, split, SEGMENT_NAMES),
        email_x[test],
        email_codes[test],
        email_y[test],
        grids,
        seed + 100,
        payload,
        args.out,
        with_qini=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
