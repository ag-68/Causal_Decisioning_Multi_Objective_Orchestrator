"""Inverse-propensity and doubly robust values, Qini curves, and the Pareto sweep."""

from __future__ import annotations

import numpy as np

from app.data import (
    ACTIONS,
    CONTROL,
    N_ACTIONS,
    SEGMENT_NAMES,
    SEGMENT_TO_CODE,
    TYPE_LOST,
    TYPE_NAMES,
    TYPE_PERSUADABLE,
    TYPE_SLEEPING,
    TYPE_SURE,
    covariate_matrix,
)
from app.models import contrasts, make_hgb
from app.optimizer import assign_heuristic, assign_random, assign_ranked, eligibility, solve_assignment


def ips_value(logged: np.ndarray, policy: np.ndarray, rewards: np.ndarray, propensity: np.ndarray) -> float:
    """Mean of ``1{A=pi(x)} / p(A|x) * R``."""
    logged = np.asarray(logged, dtype=int)
    policy = np.asarray(policy, dtype=int)
    rewards = np.asarray(rewards, dtype=np.float64)
    weight = np.take(np.asarray(propensity, dtype=np.float64), logged)
    return float(np.mean((logged == policy) * rewards / np.maximum(weight, 1e-6)))


def dr_value(
    mu: np.ndarray,
    logged: np.ndarray,
    policy: np.ndarray,
    rewards: np.ndarray,
    propensity: np.ndarray,
) -> float:
    """Doubly robust policy value from the math note."""
    logged = np.asarray(logged, dtype=int)
    policy = np.asarray(policy, dtype=int)
    rewards = np.asarray(rewards, dtype=np.float64)
    rows = np.arange(len(rewards))
    imputed = mu[rows, policy]
    logged_mean = mu[rows, logged]
    weight = np.take(np.asarray(propensity, dtype=np.float64), logged)
    correction = (logged == policy) / np.maximum(weight, 1e-6) * (rewards - logged_mean)
    return float(np.mean(imputed + correction))


def qini_curve(
    tau: np.ndarray,
    logged: np.ndarray,
    rewards: np.ndarray,
    propensity: np.ndarray,
    n_points: int = 21,
) -> dict:
    """Cumulative IPS contrast of the best active arm versus control, ranked by tau."""
    logged = np.asarray(logged, dtype=int)
    rewards = np.asarray(rewards, dtype=np.float64)
    propensity = np.asarray(propensity, dtype=np.float64)
    n = len(rewards)
    best = np.argmax(tau[:, 1:], axis=1) + 1
    score = tau[np.arange(n), best]
    phi = np.zeros(n, dtype=np.float64)
    match = logged == best
    control = logged == CONTROL
    phi[match] = rewards[match] / np.maximum(propensity[best[match]], 1e-6)
    phi[control] = -rewards[control] / max(float(propensity[CONTROL]), 1e-6)
    order = np.argsort(-score, kind="mergesort")
    cumulative = np.cumsum(phi[order]) / n
    stops = np.linspace(0, n, n_points).astype(int)
    stops[0] = 0
    stops[-1] = n
    gains = np.zeros(n_points, dtype=np.float64)
    positive = stops > 0
    gains[positive] = cumulative[stops[positive] - 1]
    fractions = stops / n
    auuc = float(np.trapezoid(gains, fractions))
    return {
        "fraction": fractions.tolist(),
        "uplift": gains.tolist(),
        "random": (fractions * gains[-1]).tolist(),
        "auuc": auuc,
    }


def _fit_logged_models(features: np.ndarray, actions: np.ndarray, outcome: np.ndarray, n_actions: int, kind: str = "tree") -> list:
    models = []
    for arm in range(n_actions):
        mask = actions == arm
        if int(mask.sum()) < 15:
            value = float(outcome[mask].mean()) if mask.any() else float(outcome.mean())
            models.append(value)
            continue
        # Validation pick for the email file. Recorded in experiments/predictor_fit.json.
        if kind == "tree":
            model = make_hgb(
                seed=arm + 3,
                max_iter=120,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=40,
                l2_regularization=0.1,
            )
            model.fit(features[mask], outcome[mask])
        else:
            from app.models import ArmModel

            model = ArmModel(kind, arm + 3)
            model.fit(features[mask], outcome[mask])
        models.append(model)
    return models


# A chance at or above the line is called a yes. The three lines sit where the
# simulator's buy chances and stop chances actually move.
CUTOFFS = (0.20, 0.35, 0.50)
# An extra at or above the line earns a message. The lines are gaps, not chances.
EXTRA_CUTOFFS = (0.0, 0.10, 0.20)


def _at_cutoff(probability: np.ndarray, answer: np.ndarray, cutoff: float) -> dict:
    called = probability >= cutoff
    positive = int(answer.sum())
    true_positive = int(np.count_nonzero(called & answer))
    called_n = int(called.sum())
    correct = true_positive + int(np.count_nonzero(~called & ~answer))
    total = int(len(answer))
    return {
        "cutoff": float(cutoff),
        "recall": float(true_positive / positive) if positive else None,
        "precision": float(true_positive / called_n) if called_n else None,
        "accuracy": float(correct / total) if total else None,
    }


def _uplift_at(predicted: np.ndarray, real: np.ndarray, cutoff: float) -> dict:
    messaged = predicted >= cutoff
    truly = real >= cutoff
    true_positive = int(np.count_nonzero(messaged & truly))
    called_n = int(messaged.sum())
    positive = int(truly.sum())
    correct = true_positive + int(np.count_nonzero(~messaged & ~truly))
    total = int(len(predicted))
    return {
        "cutoff": float(cutoff),
        "recall": float(true_positive / positive) if positive else None,
        "precision": float(true_positive / called_n) if called_n else None,
        "accuracy": float(correct / total) if total else None,
        "real_extra": float(real[messaged].mean()) if called_n else None,
    }


def uplift_report(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Score the note with the largest predicted extra.

    ``predicted`` and ``truth`` are chances for silence, email, push, and audio.
    The extra is a note's chance minus the silence chance. The chosen note is
    the one with the largest predicted extra, and the real extra is the formula
    extra of that same note.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred_extra = predicted[:, 1:] - predicted[:, :1]
    true_extra = truth[:, 1:] - truth[:, :1]
    choice = np.argmax(pred_extra, axis=1)
    rows = np.arange(len(predicted))
    chosen_pred = pred_extra[rows, choice]
    chosen_real = true_extra[rows, choice]
    return {"at_cutoff": [_uplift_at(chosen_pred, chosen_real, cutoff) for cutoff in EXTRA_CUTOFFS]}


def chance_report(probability: np.ndarray, outcome: np.ndarray, truth: np.ndarray | None = None) -> dict:
    """How a predicted chance compares with the stored answer.

    ``mean_yes`` and ``mean_no`` average the chance over the rows whose stored
    answer is yes, and over the rows whose stored answer is no.
    ``squared_error`` is the mean squared distance to the stored 0 or 1.
    ``gap_to_truth`` is the mean absolute distance to the chance that drew the
    answer. It is None when that chance is not known.
    """
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    answer = np.asarray(outcome, dtype=np.float64) >= 0.5
    positive = int(answer.sum())
    negative = int(len(answer) - positive)
    gap = None
    if truth is not None:
        gap = float(np.mean(np.abs(probability - np.asarray(truth, dtype=np.float64))))
    return {
        "mean_yes": float(probability[answer].mean()) if positive else None,
        "mean_no": float(probability[~answer].mean()) if negative else None,
        "squared_error": float(np.mean((probability - answer.astype(np.float64)) ** 2)),
        "gap_to_truth": gap,
        "n": int(len(answer)),
        "n_positive": positive,
        "base_rate": float(np.mean(answer)),
        "at_cutoff": [_at_cutoff(probability, answer, cutoff) for cutoff in CUTOFFS],
    }


def logged_arm_probability(models: list, features: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Predicted chance of the message stored on each row."""
    columns = []
    for model in models:
        if isinstance(model, float):
            columns.append(np.full(len(features), float(model)))
        else:
            columns.append(np.clip(model.predict(features), 0.0, 1.0))
    mu = np.column_stack(columns)
    rows = np.arange(len(actions))
    return mu[rows, np.asarray(actions, dtype=int)]


def _predict_logged_tau(models: list, features: np.ndarray) -> np.ndarray:
    columns = []
    for model in models:
        if isinstance(model, float):
            columns.append(np.full(len(features), model))
        else:
            columns.append(np.clip(model.predict(features), 0.0, 1.0))
    return contrasts(np.column_stack(columns))


def hillstrom_qini(train, test, kind: str = "tree") -> dict:
    """Fit on the training people. Score the held-out people."""
    train_codes = train["segment"].map(SEGMENT_TO_CODE).to_numpy(dtype=int)
    train_outcome = train["conversion"].to_numpy(dtype=np.float64)
    models = _fit_logged_models(covariate_matrix(train), train_codes, train_outcome, n_actions=3, kind=kind)
    codes = test["segment"].map(SEGMENT_TO_CODE).to_numpy(dtype=int)
    outcome = test["conversion"].to_numpy(dtype=np.float64)
    tau = _predict_logged_tau(models, covariate_matrix(test))
    counts = np.bincount(codes, minlength=3).astype(np.float64)
    propensity = counts / max(float(counts.sum()), 1.0)
    curve = qini_curve(tau, codes, outcome, propensity)
    mean_tau = tau[:, 1:].mean(axis=0)
    better = int(np.argmax(mean_tau))
    curve["arm"] = SEGMENT_NAMES[better + 1]
    curve["n"] = int(len(test))
    return curve


def pareto_sweep(
    *,
    tau_conv: np.ndarray,
    tau_fatigue: np.ndarray,
    mu_conv: np.ndarray,
    mu_opt: np.ndarray,
    greedy_prob: np.ndarray,
    logged: np.ndarray,
    y_conv: np.ndarray,
    y_opt: np.ndarray,
    recency: np.ndarray,
    propensity: np.ndarray,
    channel_cap: float,
    budgets: np.ndarray,
    seed: int = 0,
) -> dict:
    """Sweep the opt-out budget for the closed-loop policy. Other policies are single points."""
    n = len(logged)
    eligible = eligibility(np.zeros(n))
    control = np.zeros(n, dtype=int)
    base_conv = dr_value(mu_conv, logged, control, y_conv, propensity)
    base_opt = dr_value(mu_opt, logged, control, y_opt, propensity)
    rng = np.random.default_rng(seed)

    def point(name: str, actions: np.ndarray) -> dict:
        conv = dr_value(mu_conv, logged, actions, y_conv, propensity)
        opt = dr_value(mu_opt, logged, actions, y_opt, propensity)
        return {
            "name": name,
            "opt_out": opt - base_opt,
            "inc_conv": conv - base_conv,
            "volume": float(np.mean(actions != CONTROL)),
        }

    markers = [
        point("Random", assign_random(n, channel_cap, eligible, rng)),
        point("Static heuristic", assign_heuristic(recency, channel_cap, eligible)),
        point("Greedy predictive", assign_ranked(greedy_prob, channel_cap, eligible, allow_nonpositive=True)),
        point("Unconstrained uplift", assign_ranked(tau_conv, channel_cap, eligible, allow_nonpositive=False)),
    ]
    curve = []
    for budget in np.asarray(budgets, dtype=np.float64):
        actions, lam, _, _ = solve_assignment(
            tau_conv,
            tau_fatigue,
            budget=float(budget),
            channel_cap=channel_cap,
            eligible=eligible,
        )
        row = point("Closed-loop", actions)
        row["budget"] = float(budget)
        row["lambda"] = float(lam)
        curve.append(row)
    return {"closed_loop": curve, "policies": markers}


def closed_loop_marker(
    *,
    tau_conv: np.ndarray,
    tau_fatigue: np.ndarray,
    mu_conv: np.ndarray,
    mu_opt: np.ndarray,
    logged: np.ndarray,
    y_conv: np.ndarray,
    y_opt: np.ndarray,
    propensity: np.ndarray,
    channel_cap: float,
    budget: float,
    actions: np.ndarray | None = None,
) -> dict:
    """One closed-loop point at an exact opt-out cap, for a prepared case."""
    n = len(logged)
    eligible = eligibility(np.zeros(n))
    control = np.zeros(n, dtype=int)
    base_conv = dr_value(mu_conv, logged, control, y_conv, propensity)
    base_opt = dr_value(mu_opt, logged, control, y_opt, propensity)
    if actions is None:
        actions, lam, _, _ = solve_assignment(
            tau_conv,
            tau_fatigue,
            budget=float(budget),
            channel_cap=float(channel_cap),
            eligible=eligible,
        )
    else:
        actions = np.asarray(actions, dtype=int)
        lam = 0.0
    conv = dr_value(mu_conv, logged, actions, y_conv, propensity)
    opt = dr_value(mu_opt, logged, actions, y_opt, propensity)
    return {
        "name": "Closed-loop",
        "opt_out": opt - base_opt,
        "inc_conv": conv - base_conv,
        "volume": float(np.mean(actions != CONTROL)),
        "budget": float(budget),
        "lambda": float(lam),
    }


def _mix(actions: np.ndarray, mask: np.ndarray) -> dict:
    chosen = actions[mask]
    return {ACTIONS[arm]: float(np.mean(chosen == arm)) for arm in range(N_ACTIONS)}


def label_report(
    *,
    tau_conv: np.ndarray,
    tau_fatigue: np.ndarray,
    greedy_prob: np.ndarray,
    type_code: np.ndarray,
    fatigue: np.ndarray,
    budget: float,
    channel_cap: float,
    assigned=None,
) -> dict:
    """Send-mix by segment and tiredness, with the channel cap off and on.

    Unconstrained uplift scores extra buy and leaves the opt-out charge at 0.
    With the cap off, each person receives the largest extra buy. With the cap
    on, channels fill from that ranking. Charged and greedy are reported both ways.
    """
    n = len(type_code)
    eligible = eligibility(np.zeros(n))
    if assigned is None:
        charged_capped, lam, mu, _ = solve_assignment(
            tau_conv,
            tau_fatigue,
            budget=float(budget),
            channel_cap=float(channel_cap),
            eligible=eligible,
        )
        open_scores = np.asarray(tau_conv, dtype=np.float64) - float(lam) * np.asarray(tau_fatigue, dtype=np.float64)
        open_scores[:, CONTROL] = 0.0
        charged_open = np.argmax(open_scores, axis=1).astype(int)
    else:
        charged_capped = np.asarray(assigned, dtype=int)
        charged_open = charged_capped
        lam = 0.0
        mu = np.zeros(N_ACTIONS, dtype=np.float64)
    unconstrained_open = np.argmax(np.asarray(tau_conv, dtype=np.float64), axis=1).astype(int)
    unconstrained_capped = assign_ranked(tau_conv, channel_cap, eligible, allow_nonpositive=False)
    greedy_scores = np.array(greedy_prob, dtype=np.float64, copy=True)
    greedy_scores[:, CONTROL] = -1.0
    greedy_open = np.argmax(greedy_scores, axis=1).astype(int)
    greedy_capped = assign_ranked(greedy_prob, channel_cap, eligible, allow_nonpositive=True)
    codes = np.asarray(type_code, dtype=int)
    tired = np.asarray(fatigue, dtype=np.float64)
    rows = []
    bands = (("below 1.5", tired < 1.5), ("1.5 or above", tired >= 1.5))
    order = (
        (TYPE_PERSUADABLE, TYPE_NAMES[TYPE_PERSUADABLE]),
        (TYPE_SURE, TYPE_NAMES[TYPE_SURE]),
        (TYPE_LOST, TYPE_NAMES[TYPE_LOST]),
        (TYPE_SLEEPING, TYPE_NAMES[TYPE_SLEEPING]),
    )
    for code, name in order:
        for band, band_mask in bands:
            mask = (codes == code) & band_mask
            count = int(mask.sum())
            if count == 0:
                continue
            rows.append(
                {
                    "type": name,
                    "tiredness": band,
                    "n": count,
                    "open": {
                        "unconstrained": _mix(unconstrained_open, mask),
                        "charged": _mix(charged_open, mask),
                        "greedy": _mix(greedy_open, mask),
                        "match_charged_unconstrained": float(np.mean(charged_open[mask] == unconstrained_open[mask])),
                        "match_charged_greedy": float(np.mean(charged_open[mask] == greedy_open[mask])),
                    },
                    "capped": {
                        "unconstrained": _mix(unconstrained_capped, mask),
                        "charged": _mix(charged_capped, mask),
                        "greedy": _mix(greedy_capped, mask),
                        "match_charged_unconstrained": float(np.mean(charged_capped[mask] == unconstrained_capped[mask])),
                        "match_charged_greedy": float(np.mean(charged_capped[mask] == greedy_capped[mask])),
                    },
                }
            )
    return {
        "lambda": float(lam),
        "mu": [float(value) for value in mu],
        "opt_out_budget": float(budget),
        "channel_budget": float(channel_cap),
        "rows": rows,
    }
