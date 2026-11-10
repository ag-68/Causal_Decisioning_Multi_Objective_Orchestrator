"""T-learner, doubly robust CATE, greedy propensity, and the student copy.

The live policy uses the T-learner. Doubly robust CATE is fit on demand and
checked in tests. The student is a small boosted tree that copies the message
the teacher assigned at this morning's charges.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from app.data import CONTROL, F_INDEX, N_ACTIONS
from app.optimizer import enforce_caps, repair_budget, solve_assignment


class ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value, dtype=np.float64)


def make_hgb(
    seed: int = 0,
    max_iter: int = 40,
    max_depth: int = 3,
    learning_rate: float = 0.08,
    min_samples_leaf: int = 50,
    l2_regularization: float = 1.0,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=seed,
    )


class ArmModel:
    """One arm's regressor. Trees stay unscaled. SVM and MLP scale the columns first."""

    def __init__(self, kind: str, seed: int):
        self.kind = kind
        self.seed = int(seed)
        self.scaler = None
        self.model = None

    def fit(self, features: np.ndarray, outcome: np.ndarray) -> "ArmModel":
        if self.kind == "svm":
            width = min(64, max(2, len(features) - 1))
            self.scaler = StandardScaler()
            coded = self.scaler.fit_transform(features)
            self.model = Pipeline(
                [
                    ("map", Nystroem(kernel="rbf", gamma=0.05, n_components=width, random_state=self.seed)),
                    (
                        "svr",
                        LinearSVR(
                            C=1.0,
                            loss="squared_epsilon_insensitive",
                            dual=False,
                            max_iter=4000,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
        else:
            self.scaler = StandardScaler()
            coded = self.scaler.fit_transform(features)
            self.model = MLPRegressor(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=0.01,
                max_iter=25,
                batch_size=256,
                early_stopping=False,
                random_state=self.seed,
                tol=1e-4,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(coded, outcome)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        coded = features if self.scaler is None else self.scaler.transform(features)
        return self.model.predict(coded)


def _fit_arm(features: np.ndarray, outcome: np.ndarray, actions: np.ndarray, arm: int, seed: int, kind: str = "tree"):
    mask = actions == arm
    if int(mask.sum()) < 15:
        value = float(outcome[mask].mean()) if mask.any() else float(outcome.mean())
        return ConstantModel(value)
    if kind == "tree":
        model = make_hgb(seed + arm)
        model.fit(features[mask], outcome[mask])
        return model
    model = ArmModel(kind, seed + arm)
    model.fit(features[mask], outcome[mask])
    return model


def _predict_arms(models: list, features: np.ndarray) -> np.ndarray:
    columns = [np.clip(model.predict(features), 0.0, 1.0) for model in models]
    return np.column_stack(columns)


def contrasts(mu: np.ndarray) -> np.ndarray:
    tau = mu - mu[:, :1]
    tau[:, 0] = 0.0
    return tau


class UpliftModels:
    def __init__(self, conversion: list, optout: list, greedy):
        self.conversion = conversion
        self.optout = optout
        self.greedy = greedy

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        actions: np.ndarray,
        conversion: np.ndarray,
        optout: np.ndarray,
        seed: int = 0,
        kind: str = "tree",
    ) -> "UpliftModels":
        actions = np.asarray(actions, dtype=int)
        conv_models = [_fit_arm(features, conversion, actions, arm, seed, kind) for arm in range(N_ACTIONS)]
        opt_models = [_fit_arm(features, optout, actions, arm, seed + 20, kind) for arm in range(N_ACTIONS)]
        greedy = _fit_greedy(features, actions, conversion, seed + 40, kind)
        return cls(conv_models, opt_models, greedy)

    def predict_mu(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _predict_arms(self.conversion, features), _predict_arms(self.optout, features)

    def predict_tau(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_conv, mu_opt = self.predict_mu(features)
        return contrasts(mu_conv), contrasts(mu_opt)

    def predict_greedy(self, features: np.ndarray) -> np.ndarray:
        return _predict_greedy(self.greedy, features)

    def predict_greedy_grid(self, features: np.ndarray, f_grid: np.ndarray) -> np.ndarray:
        """Raw buy chances on a fatigue grid. Shape (n, grid, actions)."""
        n = len(features)
        grid = np.asarray(f_grid, dtype=np.float64)
        g = len(grid)
        packed = np.repeat(features, g, axis=0)
        packed[:, F_INDEX] = np.tile(grid, n)
        return self.predict_greedy(packed).reshape(n, g, N_ACTIONS)

    def predict_tau_grid(self, features: np.ndarray, f_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Contrasts on a fatigue grid. Returns arrays shaped (n, grid, actions)."""
        n = len(features)
        grid = np.asarray(f_grid, dtype=np.float64)
        g = len(grid)
        packed = np.repeat(features, g, axis=0)
        packed[:, F_INDEX] = np.tile(grid, n)
        tau_conv, tau_fat = self.predict_tau(packed)
        return tau_conv.reshape(n, g, N_ACTIONS), tau_fat.reshape(n, g, N_ACTIONS)


def _action_features(features: np.ndarray, arm: int) -> np.ndarray:
    one_hot = np.zeros((len(features), N_ACTIONS), dtype=np.float64)
    one_hot[:, arm] = 1.0
    return np.hstack([features, one_hot])


def _fit_greedy(features: np.ndarray, actions: np.ndarray, conversion: np.ndarray, seed: int, kind: str = "tree"):
    blocks = [_action_features(features, arm) for arm in range(N_ACTIONS)]
    # One model on the logged rows, with the observed action one-hot.
    design = np.zeros((len(features), features.shape[1] + N_ACTIONS), dtype=np.float64)
    for arm in range(N_ACTIONS):
        mask = actions == arm
        design[mask] = blocks[arm][mask]
    if len(features) < 15:
        return ConstantModel(float(np.mean(conversion)))
    if kind == "tree":
        model = make_hgb(seed)
        model.fit(design, conversion)
        return model
    model = ArmModel(kind, seed)
    model.fit(design, conversion)
    return model


def _predict_greedy(model, features: np.ndarray) -> np.ndarray:
    columns = []
    for arm in range(N_ACTIONS):
        pred = model.predict(_action_features(features, arm))
        columns.append(np.clip(pred, 0.0, 1.0))
    return np.column_stack(columns)


def fit_dr_cate(
    features: np.ndarray,
    actions: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    seed: int = 0,
) -> list:
    """Two-fold cross-fit doubly robust arm means.

    The pseudo-outcome for arm ``a`` is
    ``mu_a(x) + 1{A=a} / p(a) * (Y - mu_a(x))``, with ``mu_a`` fit on the
    other fold. A second regressor maps features to that pseudo-outcome.
    """
    features = np.asarray(features, dtype=np.float64)
    actions = np.asarray(actions, dtype=int)
    outcome = np.asarray(outcome, dtype=np.float64)
    propensity = np.asarray(propensity, dtype=np.float64)
    n = len(outcome)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold = np.zeros(n, dtype=int)
    fold[order[n // 2 :]] = 1
    phi = np.zeros((n, N_ACTIONS), dtype=np.float64)
    for held in (0, 1):
        train = fold != held
        test = fold == held
        arm_models = [_fit_arm(features, outcome, actions, arm, seed + 10 * held) for arm in range(N_ACTIONS)]
        mu = _predict_arms(arm_models, features[test])
        logged = actions[test]
        correction = np.zeros_like(mu)
        for arm in range(N_ACTIONS):
            hit = logged == arm
            p = max(float(propensity[arm]), 1e-3)
            correction[hit, arm] = (outcome[test][hit] - mu[hit, arm]) / p
        phi[test] = mu + correction
    second = []
    for arm in range(N_ACTIONS):
        model = make_hgb(seed + 100 + arm)
        model.fit(features, phi[:, arm])
        second.append(model)
    return second


def predict_dr_tau(models: list, features: np.ndarray) -> np.ndarray:
    return contrasts(_predict_arms(models, features))


def _messages_at_charges(
    tau_conv: np.ndarray,
    tau_fatigue: np.ndarray,
    lam: float,
    mu: np.ndarray,
    budget: float,
    channel_cap: float,
) -> np.ndarray:
    """Score at frozen charges, apply the channel cap, then the opt-out repair."""
    n, k = tau_conv.shape
    scores = tau_conv - float(lam) * tau_fatigue - np.asarray(mu, dtype=np.float64).reshape(1, -1)
    scores[:, CONTROL] = 0.0
    eligible = np.ones((n, k), dtype=bool)
    caps = np.full(k, float(channel_cap), dtype=np.float64)
    caps[CONTROL] = 1.0
    actions = enforce_caps(scores, eligible, caps)
    return repair_budget(actions, scores, tau_fatigue, budget)


def _batch_ms(fn, repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1000.0


def fit_student(
    uplift: UpliftModels,
    x_train: np.ndarray,
    budget: float,
    channel_cap: float,
    epochs: int = 40,
    seed: int = 0,
) -> HistGradientBoostingClassifier:
    """Small tree that copies the messages the tree rule assigns at these caps."""
    tau_c, tau_f = uplift.predict_tau(x_train)
    _, lam, mu, _ = solve_assignment(
        tau_c,
        tau_f,
        budget=budget,
        channel_cap=channel_cap,
        eligible=np.ones((len(x_train), N_ACTIONS), dtype=bool),
    )
    labels = _messages_at_charges(tau_c, tau_f, lam, mu, budget, channel_cap)
    model = HistGradientBoostingClassifier(
        max_iter=max(int(epochs), 1),
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x_train, labels)
    return model


def student_messages(
    model,
    features: np.ndarray,
    fatigue: np.ndarray,
    channel_cap: float,
    eligible: np.ndarray,
) -> np.ndarray:
    """The student's message, then the channel cap and the weekly skip."""
    from app.data import F_INDEX

    frame = np.array(features, dtype=np.float64, copy=True)
    frame[:, F_INDEX] = np.asarray(fatigue, dtype=np.float64)
    raw = model.predict(frame).astype(int)
    n = len(raw)
    scores = np.full((n, N_ACTIONS), -1.0, dtype=np.float64)
    scores[:, CONTROL] = 0.0
    scores[np.arange(n), raw] = 1.0
    caps = np.full(N_ACTIONS, float(channel_cap), dtype=np.float64)
    caps[CONTROL] = 1.0
    return enforce_caps(scores, eligible, caps)


def distill_policy(
    uplift: UpliftModels,
    x_train: np.ndarray,
    x_test: np.ndarray,
    budget: float,
    channel_cap: float,
    epochs: int = 40,
    seed: int = 0,
) -> dict:
    """Train a small boosted tree to copy the teacher's message. Report agreement and latency.

    The labels are the messages assigned at the training morning's charges.
    The held-out people are graded at those same charges.
    """
    tau_c_tr, tau_f_tr = uplift.predict_tau(x_train)
    tau_c_te, tau_f_te = uplift.predict_tau(x_test)
    _, lam, mu, _ = solve_assignment(
        tau_c_tr,
        tau_f_tr,
        budget=budget,
        channel_cap=channel_cap,
        eligible=np.ones((len(x_train), N_ACTIONS), dtype=bool),
    )
    train_actions = _messages_at_charges(tau_c_tr, tau_f_tr, lam, mu, budget, channel_cap)
    teacher = _messages_at_charges(tau_c_te, tau_f_te, lam, mu, budget, channel_cap)
    model = HistGradientBoostingClassifier(
        max_iter=max(int(epochs), 1),
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x_train, train_actions)
    student_train = model.predict(x_train).astype(int)
    student = model.predict(x_test).astype(int)
    train_fidelity = float(np.mean(student_train == train_actions))
    fidelity = float(np.mean(student == teacher))

    latency_n = min(1024, len(x_test))
    x_np = x_test[:latency_n]

    def student_forward():
        model.predict(x_np)

    def teacher_forward():
        tau_c, tau_f = uplift.predict_tau(x_np)
        _messages_at_charges(tau_c, tau_f, lam, mu, budget, channel_cap)

    return {
        "fidelity": fidelity,
        "train_fidelity": train_fidelity,
        "lambda": float(lam),
        "student_batch_ms": _batch_ms(student_forward, repeats=20),
        "teacher_batch_ms": _batch_ms(teacher_forward, repeats=5),
        "batch_size": latency_n,
        "epochs": int(epochs),
    }
