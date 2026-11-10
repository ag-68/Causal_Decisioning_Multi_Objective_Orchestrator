"""Primal-dual channel assignment and the multi-day fatigue loop.

Given prices, each user takes the action with the largest score

    tau_conv - lambda * tau_fatigue - mu_a

and Control scores 0. Channel caps are a hard repair. A final repair keeps
predicted incremental opt-out inside the budget. The static-alpha policy
freezes the day-0 price and does not see the weekend multiplier on tau_fatigue.
"""

from __future__ import annotations

import numpy as np

from app.data import (
    AUDIO,
    CONTROL,
    EMAIL,
    N_ACTIONS,
    PUSH,
    WEEKLY_TOUCHES,
    conversion_prob,
    fatigue_step,
    optout_prob,
)


def enforce_caps(scores: np.ndarray, eligible: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Argmax assignment with hard per-channel share caps. Control is uncapped."""
    n, k = scores.shape
    adjusted = np.array(scores, dtype=np.float64, copy=True)
    adjusted[~eligible] = -1e9
    adjusted[:, CONTROL] = 0.0
    actions = np.argmax(adjusted, axis=1).astype(int)
    limits = np.maximum(np.floor(np.asarray(caps, dtype=np.float64) * n).astype(int), 0)
    limits[CONTROL] = n
    for _ in range(k):
        overflow = False
        for arm in range(1, k):
            chosen = np.flatnonzero(actions == arm)
            extra = len(chosen) - int(limits[arm])
            if extra <= 0:
                continue
            overflow = True
            alternatives = adjusted[chosen].copy()
            alternatives[:, arm] = -1e9
            second = np.argmax(alternatives, axis=1)
            gap = adjusted[chosen, arm] - alternatives[np.arange(len(chosen)), second]
            drop_local = np.argsort(gap, kind="mergesort")[:extra]
            actions[chosen[drop_local]] = second[drop_local]
        if not overflow:
            break
    return actions


def repair_budget(
    actions: np.ndarray,
    scores: np.ndarray,
    tau_fatigue: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Drop the lowest-score costly sends until mean predicted opt-out is within budget."""
    actions = np.array(actions, copy=True)
    n = len(actions)
    cost = tau_fatigue[np.arange(n), actions]
    excess = float(cost.sum() - budget * n)
    if excess <= 1e-9:
        return actions
    treated = np.flatnonzero(actions != CONTROL)
    if len(treated) == 0:
        return actions
    chosen_score = scores[np.arange(len(actions)), actions]
    order = treated[np.argsort(chosen_score[treated], kind="mergesort")]
    contrib = tau_fatigue[order, actions[order]]
    keep = contrib > 0.0
    order = order[keep]
    contrib = contrib[keep]
    if len(order) == 0:
        return actions
    n_drop = int(np.searchsorted(np.cumsum(contrib), excess, side="left") + 1)
    actions[order[: min(n_drop, len(order))]] = CONTROL
    return actions


def solve_assignment(
    tau_conv: np.ndarray,
    tau_fatigue: np.ndarray,
    budget: float,
    channel_cap: float,
    eligible: np.ndarray,
    n_steps: int = 80,
    eta_lambda: float = 2.0,
    eta_mu: float = 0.12,
    lambda_init: float = 0.0,
    mu_init: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Projected dual steps, then a hard budget repair. Returns actions, lambda, mu, scores."""
    n, k = tau_conv.shape
    caps = np.full(k, float(channel_cap), dtype=np.float64)
    caps[CONTROL] = 1.0
    lam = float(lambda_init)
    mu = np.zeros(k, dtype=np.float64) if mu_init is None else np.array(mu_init, dtype=np.float64, copy=True)
    mu[CONTROL] = 0.0
    actions = np.zeros(n, dtype=int)
    scores = np.zeros((n, k), dtype=np.float64)
    for _ in range(int(n_steps)):
        scores = tau_conv - lam * tau_fatigue - mu.reshape(1, -1)
        scores[:, CONTROL] = 0.0
        actions = enforce_caps(scores, eligible, caps)
        predicted = float(tau_fatigue[np.arange(n), actions].mean())
        shares = np.bincount(actions, minlength=k) / n
        lam = max(0.0, lam + eta_lambda * (predicted - budget))
        for arm in range(1, k):
            mu[arm] = max(0.0, mu[arm] + eta_mu * (float(shares[arm]) - channel_cap))
        mu[CONTROL] = 0.0
    scores = tau_conv - lam * tau_fatigue - mu.reshape(1, -1)
    scores[:, CONTROL] = 0.0
    actions = enforce_caps(scores, eligible, caps)
    actions = repair_budget(actions, scores, tau_fatigue, budget)
    return actions, lam, mu, scores


def assign_ranked(
    action_scores: np.ndarray,
    channel_cap: float,
    eligible: np.ndarray,
    allow_nonpositive: bool,
) -> np.ndarray:
    """Fill channel caps from the highest score down. Control is the leftover."""
    n, k = action_scores.shape
    active = np.array(action_scores, dtype=np.float64, copy=True)
    active[:, CONTROL] = -1.0
    active[~eligible] = -1.0
    best = np.argmax(active, axis=1)
    best_score = active[np.arange(n), best]
    order = np.argsort(-best_score, kind="mergesort")
    actions = np.zeros(n, dtype=int)
    limits = np.full(k, int(np.floor(channel_cap * n)), dtype=int)
    counts = np.zeros(k, dtype=int)
    for index in order:
        if not allow_nonpositive and best_score[index] <= 0.0:
            break
        ranking = np.argsort(-active[index])
        for arm in ranking:
            arm = int(arm)
            if arm == CONTROL or not eligible[index, arm]:
                continue
            if counts[arm] >= limits[arm]:
                continue
            actions[index] = arm
            counts[arm] += 1
            break
    return actions


def assign_heuristic(recency: np.ndarray, channel_cap: float, eligible: np.ndarray) -> np.ndarray:
    """Push when months since purchase are at least 3. Otherwise Email. Caps still apply."""
    n = len(recency)
    scores = np.full((n, N_ACTIONS), -1.0, dtype=np.float64)
    scores[:, CONTROL] = 0.0
    preferred = np.where(np.asarray(recency) >= 3.0, PUSH, EMAIL)
    scores[np.arange(n), preferred] = 1.0
    caps = np.full(N_ACTIONS, float(channel_cap), dtype=np.float64)
    caps[CONTROL] = 1.0
    return enforce_caps(scores, eligible, caps)


def assign_random(n: int, channel_cap: float, eligible: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    actions = np.zeros(n, dtype=int)
    limits = np.full(N_ACTIONS, int(np.floor(channel_cap * n)), dtype=int)
    counts = np.zeros(N_ACTIONS, dtype=int)
    for index in rng.permutation(n):
        choices = [arm for arm in (EMAIL, PUSH, AUDIO) if eligible[index, arm] and counts[arm] < limits[arm]]
        if not choices:
            continue
        arm = int(rng.choice(choices))
        actions[index] = arm
        counts[arm] += 1
    return actions


def eligibility(recent_touches: np.ndarray, weekly_cap: int = WEEKLY_TOUCHES) -> np.ndarray:
    n = len(recent_touches)
    allowed = np.ones((n, N_ACTIONS), dtype=bool)
    allowed[np.asarray(recent_touches) >= weekly_cap, 1:] = False
    return allowed


def interpolate_tau(f_grid: np.ndarray, tau_grid: np.ndarray, fatigue: np.ndarray) -> np.ndarray:
    """Linear interpolation along the fatigue axis of an (n, grid, actions) tensor."""
    grid = np.asarray(f_grid, dtype=np.float64)
    fatigue = np.clip(np.asarray(fatigue, dtype=np.float64), grid[0], grid[-1])
    index = np.searchsorted(grid, fatigue, side="right") - 1
    index = np.clip(index, 0, len(grid) - 2)
    span = grid[index + 1] - grid[index]
    weight = np.clip((fatigue - grid[index]) / span, 0.0, 1.0)
    rows = np.arange(len(fatigue))
    left = tau_grid[rows, index]
    right = tau_grid[rows, index + 1]
    return (1.0 - weight)[:, None] * left + weight[:, None] * right


def _paired(p_action: np.ndarray, p_control: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    draw = rng.random(len(p_action))
    return (draw < p_action).astype(np.float64), (draw < p_control).astype(np.float64)


def simulate_horizon(
    *,
    baseline: np.ndarray,
    type_code: np.ndarray,
    tau_conv_grid: np.ndarray,
    tau_fat_grid: np.ndarray,
    f_grid: np.ndarray,
    budget: float,
    channel_cap: float,
    traffic_shift: float,
    horizon: int,
    seed: int,
    weekly_cap: int = WEEKLY_TOUCHES,
    closed_policy=None,
) -> dict:
    """Closed-loop versus frozen-alpha over ``horizon`` days, both facing the weekend shift."""
    rng = np.random.default_rng(seed)
    n = len(baseline)
    horizon = int(horizon)
    f_closed = np.zeros(n, dtype=np.float64)
    f_static = np.zeros(n, dtype=np.float64)
    sent_closed = np.zeros((n, horizon), dtype=np.int8)
    sent_static = np.zeros((n, horizon), dtype=np.int8)
    lam = 0.0
    mu = np.zeros(N_ACTIONS, dtype=np.float64)
    alpha = 0.0
    days: list[int] = []
    lambdas: list[float] = []
    opt_closed: list[float] = []
    opt_static: list[float] = []
    inc_closed: list[float] = []
    inc_static: list[float] = []
    weekends: list[bool] = []
    mix = np.zeros(N_ACTIONS, dtype=np.float64)
    volume = []

    zero = np.zeros(n, dtype=int)
    for day in range(horizon):
        weekend = (day % 7) >= 5
        shift = float(traffic_shift) if weekend else 1.0
        recent_c = sent_closed[:, max(0, day - 7) : day].sum(axis=1) if day else np.zeros(n)
        recent_s = sent_static[:, max(0, day - 7) : day].sum(axis=1) if day else np.zeros(n)
        if closed_policy is None:
            tau_c = interpolate_tau(f_grid, tau_conv_grid, f_closed)
            tau_f = interpolate_tau(f_grid, tau_fat_grid, f_closed)
            actions, lam, mu, _ = solve_assignment(
                tau_c,
                tau_f * shift,
                budget=budget,
                channel_cap=channel_cap,
                eligible=eligibility(recent_c, weekly_cap),
                lambda_init=lam,
                mu_init=mu,
            )
        else:
            actions = np.asarray(closed_policy(f_closed, recent_c), dtype=int)
            lam = 0.0
            mu = np.zeros(N_ACTIONS, dtype=np.float64)
        if day == 0:
            alpha = lam
            actions_static = actions.copy()
            preferred_static = actions.copy()
        elif closed_policy is None:
            tau_c_s = interpolate_tau(f_grid, tau_conv_grid, f_static)
            tau_f_s = interpolate_tau(f_grid, tau_fat_grid, f_static)
            scores = tau_c_s - alpha * tau_f_s
            scores[:, CONTROL] = 0.0
            caps = np.full(N_ACTIONS, float(channel_cap))
            caps[CONTROL] = 1.0
            actions_static = enforce_caps(scores, eligibility(recent_s, weekly_cap), caps)
        else:
            scores = np.full((n, N_ACTIONS), -1.0, dtype=np.float64)
            scores[:, CONTROL] = 0.0
            scores[np.arange(n), preferred_static] = 1.0
            caps = np.full(N_ACTIONS, float(channel_cap))
            caps[CONTROL] = 1.0
            actions_static = enforce_caps(scores, eligibility(recent_s, weekly_cap), caps)

        p_conv = conversion_prob(baseline, type_code, actions, f_closed)
        p_base = conversion_prob(baseline, type_code, zero, f_closed)
        y, y0 = _paired(p_conv, p_base, rng)
        p_opt = optout_prob(actions, f_closed, shift=shift)
        p_opt0 = optout_prob(zero, f_closed, shift=shift)
        opt, opt0 = _paired(p_opt, p_opt0, rng)

        p_conv_s = conversion_prob(baseline, type_code, actions_static, f_static)
        p_base_s = conversion_prob(baseline, type_code, zero, f_static)
        y_s, y0_s = _paired(p_conv_s, p_base_s, rng)
        p_opt_s = optout_prob(actions_static, f_static, shift=shift)
        p_opt0_s = optout_prob(zero, f_static, shift=shift)
        opt_s, opt0_s = _paired(p_opt_s, p_opt0_s, rng)

        days.append(day)
        lambdas.append(float(lam))
        weekends.append(weekend)
        inc_closed.append(float(np.mean(y - y0)))
        inc_static.append(float(np.mean(y_s - y0_s)))
        opt_closed.append(float(np.mean(opt - opt0)))
        opt_static.append(float(np.mean(opt_s - opt0_s)))
        volume.append(float(np.mean(actions != CONTROL)))
        mix += np.bincount(actions, minlength=N_ACTIONS) / n

        sent_closed[:, day] = (actions != CONTROL).astype(np.int8)
        sent_static[:, day] = (actions_static != CONTROL).astype(np.int8)
        f_closed = np.maximum(fatigue_step(f_closed, actions, rng=rng), 0.0)
        f_static = np.maximum(fatigue_step(f_static, actions_static, rng=rng), 0.0)

    from app.data import ACTIONS

    return {
        "days": days,
        "weekend": weekends,
        "lambda": lambdas,
        "alpha": float(alpha),
        "opt_out_closed": opt_closed,
        "opt_out_static": opt_static,
        "inc_conv_closed": inc_closed,
        "inc_conv_static": inc_static,
        "channel_mix": {ACTIONS[arm]: float(mix[arm] / horizon) for arm in range(N_ACTIONS)},
        "summary": {
            "mean_inc_conv_closed": float(np.mean(inc_closed)),
            "mean_inc_conv_static": float(np.mean(inc_static)),
            "mean_opt_out_closed": float(np.mean(opt_closed)),
            "mean_opt_out_static": float(np.mean(opt_static)),
            "mean_volume_closed": float(np.mean(volume)),
            "final_lambda": float(lam),
            "alpha": float(alpha),
        },
    }


def _day_gaps(
    actions: np.ndarray,
    fatigue: np.ndarray,
    baseline: np.ndarray,
    type_code: np.ndarray,
    shift: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Extra buy and extra stop for one morning, from one shared coin against silence."""
    zero = np.zeros(len(actions), dtype=int)
    bought, bought_silent = _paired(
        conversion_prob(baseline, type_code, actions, fatigue),
        conversion_prob(baseline, type_code, zero, fatigue),
        rng,
    )
    stopped, stopped_silent = _paired(
        optout_prob(actions, fatigue, shift=shift),
        optout_prob(zero, fatigue, shift=shift),
        rng,
    )
    return float(np.mean(bought - bought_silent)), float(np.mean(stopped - stopped_silent))


def _assign_horizon_rule(
    name: str,
    *,
    fatigue: np.ndarray,
    recent: np.ndarray,
    recency: np.ndarray,
    tau_conv_grid: np.ndarray,
    tau_fat_grid: np.ndarray,
    greedy_grid: np.ndarray,
    f_grid: np.ndarray,
    channel_cap: float,
    budget: float,
    shift: float,
    weekly_cap: int,
    lam: float,
    mu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray]:
    eligible = eligibility(recent, weekly_cap)
    if name == "Closed-loop":
        tau_c = interpolate_tau(f_grid, tau_conv_grid, fatigue)
        tau_f = interpolate_tau(f_grid, tau_fat_grid, fatigue)
        return solve_assignment(
            tau_c,
            tau_f * shift,
            budget=budget,
            channel_cap=channel_cap,
            eligible=eligible,
            lambda_init=lam,
            mu_init=mu,
        )[:3]
    if name == "Random":
        return assign_random(len(fatigue), channel_cap, eligible, rng), lam, mu
    if name == "Static heuristic":
        return assign_heuristic(recency, channel_cap, eligible), lam, mu
    if name == "Greedy predictive":
        scores = interpolate_tau(f_grid, greedy_grid, fatigue)
        return assign_ranked(scores, channel_cap, eligible, allow_nonpositive=True), lam, mu
    scores = interpolate_tau(f_grid, tau_conv_grid, fatigue)
    return assign_ranked(scores, channel_cap, eligible, allow_nonpositive=False), lam, mu


def horizon_pareto(
    *,
    baseline: np.ndarray,
    type_code: np.ndarray,
    recency: np.ndarray,
    tau_conv_grid: np.ndarray,
    tau_fat_grid: np.ndarray,
    greedy_grid: np.ndarray,
    f_grid: np.ndarray,
    channel_cap: float,
    budgets: np.ndarray,
    traffic_shift: float,
    horizon: int,
    seed: int,
    weekly_cap: int = WEEKLY_TOUCHES,
) -> dict:
    """Mean extra buy and extra stop over ``horizon`` mornings, one tiredness path per rule.

    Closed-loop is one path per opt-out cap. The charge starts at zero and is carried
    into the next morning. The other rules ignore the opt-out cap. Every rule faces
    the weekly touch limit and the weekend shift on the stop coin.
    """
    n = len(baseline)
    horizon = int(horizon)

    def run(name: str, budget: float, rule_seed: int) -> dict:
        rng = np.random.default_rng(rule_seed)
        fatigue = np.zeros(n, dtype=np.float64)
        sent = np.zeros((n, horizon), dtype=np.int8)
        lam = 0.0
        mu = np.zeros(N_ACTIONS, dtype=np.float64)
        buys: list[float] = []
        stops: list[float] = []
        for day in range(horizon):
            shift = float(traffic_shift) if (day % 7) >= 5 else 1.0
            recent = sent[:, max(0, day - 7) : day].sum(axis=1) if day else np.zeros(n)
            actions, lam, mu = _assign_horizon_rule(
                name,
                fatigue=fatigue,
                recent=recent,
                recency=recency,
                tau_conv_grid=tau_conv_grid,
                tau_fat_grid=tau_fat_grid,
                greedy_grid=greedy_grid,
                f_grid=f_grid,
                channel_cap=channel_cap,
                budget=float(budget),
                shift=shift,
                weekly_cap=weekly_cap,
                lam=lam,
                mu=mu,
                rng=rng,
            )
            buy, stop = _day_gaps(actions, fatigue, baseline, type_code, shift, rng)
            buys.append(buy)
            stops.append(stop)
            sent[:, day] = (actions != CONTROL).astype(np.int8)
            fatigue = np.maximum(fatigue_step(fatigue, actions, rng=rng), 0.0)
        row = {
            "name": name,
            "opt_out": float(np.mean(stops)),
            "inc_conv": float(np.mean(buys)),
            "lambda": float(lam),
        }
        if name == "Closed-loop":
            row["budget"] = float(budget)
        return row

    closed = [run("Closed-loop", float(budget), seed + 20 + index) for index, budget in enumerate(np.asarray(budgets, dtype=np.float64))]
    others = [
        run(name, 0.0, seed + 40 + index)
        for index, name in enumerate(("Random", "Static heuristic", "Greedy predictive", "Unconstrained uplift"))
    ]
    return {"closed_loop": closed, "policies": others}
