"""FastAPI surface for the closed-loop orchestrator."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.data import (
    ACTIONS,
    AUDIO,
    CONTROL,
    N_ACTIONS,
    PUSH,
    TYPE_NAMES,
    TYPE_PERSUADABLE,
    TYPE_SLEEPING,
    TYPE_SURE,
    bitrate_illustration,
    SPLIT_TEST,
    WEEKLY_TOUCHES,
    conversion_prob,
    feature_matrix,
    load_hillstrom,
    make_cross_section,
    optout_prob,
    prepare_demo,
)
from app.models import UpliftModels, distill_policy, fit_student, student_messages
from app.ope import (
    chance_report,
    closed_loop_marker,
    uplift_report,
    hillstrom_qini,
    label_report,
    pareto_sweep,
    qini_curve,
)
from app.optimizer import assign_ranked, eligibility, horizon_pareto, simulate_horizon, solve_assignment

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "static" / "index.html"
FORMULATION = ROOT / "docs" / "mathematical_formulation.md"

NOTE_SIM = (
    "Pareto, shadow prices, the Qini curve, and the label tables use the "
    "declared four-channel fatigue simulator on Hillstrom shopping cards."
)

PREPARED_CASES = (
    ("Tight opt-out, tight channel", 0.015, 0.15),
    ("Tight opt-out, relaxed channel", 0.015, 0.30),
    ("Relaxed opt-out, tight channel", 0.05, 0.15),
    ("Relaxed opt-out, relaxed channel", 0.05, 0.30),
)
PREPARED_SHIFT = 1.8
PREPARED_HORIZON = 21
PARETO_OPT_BUDGETS = np.array([0.005, 0.01, 0.025, 0.05])
PARETO_POLICY_ORDER = (
    "Random",
    "Static heuristic",
    "Greedy predictive",
    "Unconstrained uplift",
)
NOTE_FALLBACK = (
    "The shopping-card file was not loaded. The cards are a seeded stand-in "
    "with the same columns. The charts come from the declared fatigue simulator."
)

app = FastAPI(title="Closed-loop message orchestrator")
_session = None
_lock = threading.Lock()


class SimRequest(BaseModel):
    opt_out_budget: float = 0.015
    channel_budget: float = 0.30
    traffic_shift: float = 1.8
    horizon_days: int = 21
    predictor: str = "tree"


class ParetoRequest(BaseModel):
    channel_budget: float = 0.30
    opt_out_budget: float | None = None
    predictor: str = "tree"


class DistillRequest(BaseModel):
    opt_out_budget: float = 0.015
    channel_budget: float = 0.30
    epochs: int = Field(default=40, ge=1, le=200)


class InspectRequest(BaseModel):
    opt_out_budget: float = 0.015
    channel_budget: float = 0.30
    predictor: str = "tree"


PREDICTORS = {"tree", "svm", "mlp", "student"}


def _kind(name: str) -> str:
    return name if name in PREDICTORS else "tree"


def reset_state() -> None:
    global _session
    with _lock:
        _session = None


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def build_session():
    fallback = os.environ.get("ORCH_FALLBACK", "0") == "1"
    n = int(os.environ.get("ORCH_N", "0"))
    seed = int(os.environ.get("ORCH_SEED", "0"))
    full, source = load_hillstrom(n_fallback=n if n > 0 else 8000, seed=seed, fallback=fallback)
    demo = prepare_demo(full, n=n, seed=seed)
    cross = make_cross_section(demo, seed=seed)
    # Validation rows join the fit once settings are chosen. The test stays out.
    train = cross.loc[cross["split"] != SPLIT_TEST].reset_index(drop=True)
    test = cross.loc[cross["split"] == SPLIT_TEST].reset_index(drop=True)
    x_train = feature_matrix(train, train["f"].to_numpy(np.float64))
    x_test = feature_matrix(test, test["f"].to_numpy(np.float64))
    uplift = UpliftModels.fit(
        x_train,
        train["action"].to_numpy(int),
        train["y_conversion"].to_numpy(np.float64),
        train["y_optout"].to_numpy(np.float64),
        seed=seed,
    )
    mu_conv, mu_opt = uplift.predict_mu(x_test)
    tau_conv, tau_fat = uplift.predict_tau(x_test)
    f_grid = np.linspace(0.0, 4.0, 17)
    tau_conv_grid, tau_fat_grid = uplift.predict_tau_grid(x_test, f_grid)
    propensity = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
    return {
        "source": source,
        "seed": seed,
        "note": NOTE_FALLBACK if source == "fallback" else NOTE_SIM,
        "train": train,
        "test": test,
        "x_train": x_train,
        "x_test": x_test,
        "uplift": uplift,
        "mu_conv": mu_conv,
        "mu_opt": mu_opt,
        "tau_conv": tau_conv,
        "tau_fat": tau_fat,
        "greedy_prob": uplift.predict_greedy(x_test),
        "f_grid": f_grid,
        "tau_conv_grid": tau_conv_grid,
        "tau_fat_grid": tau_fat_grid,
        "propensity": propensity,
        "qini_sim": qini_curve(
            tau_conv,
            test["action"].to_numpy(int),
            test["y_conversion"].to_numpy(np.float64),
            propensity,
        ),
        "qini_hillstrom": hillstrom_qini(train, test),
    }


def _fit_view(session: dict, kind: str) -> dict:
    train = session["train"]
    uplift = UpliftModels.fit(
        session["x_train"],
        train["action"].to_numpy(int),
        train["y_conversion"].to_numpy(np.float64),
        train["y_optout"].to_numpy(np.float64),
        seed=session["seed"],
        kind=kind,
    )
    test = session["test"]
    x_test = session["x_test"]
    mu_conv, mu_opt = uplift.predict_mu(x_test)
    tau_conv, tau_fat = uplift.predict_tau(x_test)
    tau_conv_grid, tau_fat_grid = uplift.predict_tau_grid(x_test, session["f_grid"])
    greedy = uplift.predict_greedy(x_test)
    return {
        "tau_conv": tau_conv,
        "tau_fat": tau_fat,
        "mu_conv": mu_conv,
        "mu_opt": mu_opt,
        "greedy_prob": greedy,
        "tau_conv_grid": tau_conv_grid,
        "tau_fat_grid": tau_fat_grid,
        "qini_sim": qini_curve(
            tau_conv,
            test["action"].to_numpy(int),
            test["y_conversion"].to_numpy(np.float64),
            session["propensity"],
        ),
        "qini_hillstrom": hillstrom_qini(train, test, kind=kind),
    }


def _view(session: dict, kind: str) -> dict:
    """Trees are already on the session. SVM and MLP are fit once, then reused."""
    if kind in {"tree", "student"}:
        return session
    with _lock:
        cache = session.setdefault("predictor_cache", {})
        if kind not in cache:
            cache[kind] = _fit_view(session, kind)
        return cache[kind]


def _student_model(session: dict, budget: float, channel_cap: float):
    key = (round(float(budget), 4), round(float(channel_cap), 4))
    with _lock:
        cache = session.setdefault("student_cache", {})
        if key not in cache:
            cache[key] = fit_student(
                session["uplift"],
                session["x_train"],
                budget,
                channel_cap,
                seed=session["seed"],
            )
        return cache[key]


def _student_at(session: dict, budget: float, channel_cap: float, fatigue: np.ndarray, recent: np.ndarray) -> np.ndarray:
    model = _student_model(session, budget, channel_cap)
    return student_messages(
        model,
        session["x_test"],
        fatigue,
        channel_cap,
        eligibility(recent),
    )


def _simulate(session: dict, budget: float, channel_cap: float, shift: float, horizon: int, predictor: str = "tree") -> dict:
    test = session["test"]
    kind = _kind(predictor)
    view = _view(session, kind)
    closed_policy = None
    if kind == "student":
        def closed_policy(fatigue, recent, budget=budget, channel_cap=channel_cap):
            return _student_at(session, budget, channel_cap, fatigue, recent)
    result = simulate_horizon(
        baseline=test["baseline"].to_numpy(np.float64),
        type_code=test["type_code"].to_numpy(int),
        tau_conv_grid=view["tau_conv_grid"],
        tau_fat_grid=view["tau_fat_grid"],
        f_grid=session["f_grid"],
        budget=budget,
        channel_cap=channel_cap,
        traffic_shift=shift,
        horizon=horizon,
        seed=session["seed"] + 3,
        closed_policy=closed_policy,
    )
    result["source"] = session["source"]
    result["note"] = session["note"]
    result["budget"] = budget
    return result


def _labels(session: dict, budget: float, channel_cap: float, predictor: str = "tree") -> dict:
    test = session["test"]
    kind = _kind(predictor)
    view = _view(session, kind)
    assigned = None
    if kind == "student":
        assigned = _student_at(
            session,
            budget,
            channel_cap,
            test["f"].to_numpy(np.float64),
            np.zeros(len(test)),
        )
    return label_report(
        tau_conv=view["tau_conv"],
        tau_fatigue=view["tau_fat"],
        greedy_prob=view["greedy_prob"],
        type_code=test["type_code"].to_numpy(int),
        fatigue=test["f"].to_numpy(np.float64),
        budget=budget,
        channel_cap=channel_cap,
        assigned=assigned,
    )


def _pareto(session: dict, channel_cap: float, mark_budget: float | None, predictor: str = "tree") -> dict:
    test = session["test"]
    kind = _kind(predictor)
    view = _view(session, kind)
    swept = pareto_sweep(
        tau_conv=view["tau_conv"],
        tau_fatigue=view["tau_fat"],
        mu_conv=view["mu_conv"],
        mu_opt=view["mu_opt"],
        greedy_prob=view["greedy_prob"],
        logged=test["action"].to_numpy(int),
        y_conv=test["y_conversion"].to_numpy(np.float64),
        y_opt=test["y_optout"].to_numpy(np.float64),
        recency=test["recency"].to_numpy(np.float64),
        propensity=session["propensity"],
        channel_cap=channel_cap,
        budgets=np.linspace(0.005, 0.06, 8),
        seed=session["seed"],
    )
    if mark_budget is not None:
        student_actions = None
        if kind == "student":
            student_actions = _student_at(
                session,
                mark_budget,
                channel_cap,
                test["f"].to_numpy(np.float64),
                np.zeros(len(test)),
            )
        swept["marked"] = closed_loop_marker(
            tau_conv=view["tau_conv"],
            tau_fatigue=view["tau_fat"],
            mu_conv=view["mu_conv"],
            mu_opt=view["mu_opt"],
            logged=test["action"].to_numpy(int),
            y_conv=test["y_conversion"].to_numpy(np.float64),
            y_opt=test["y_optout"].to_numpy(np.float64),
            propensity=session["propensity"],
            channel_cap=channel_cap,
            budget=mark_budget,
            actions=student_actions,
        )
        if kind == "student":
            swept["closed_loop"] = [swept["marked"]]
    swept["source"] = session["source"]
    swept["note"] = session["note"]
    swept["qini_sim"] = view["qini_sim"]
    swept["qini_hillstrom"] = view["qini_hillstrom"]
    return swept


def _levels(rows: list[dict], key: str) -> list[dict]:
    values = [float(row[key]) for row in rows]
    low, high = min(values), max(values)
    span = high - low or 1.0
    for row, value in zip(rows, values):
        row["level"] = (value - low) / span
    return rows


def _pareto_kwargs(session: dict, channel_cap: float, budgets: np.ndarray) -> dict:
    test = session["test"]
    return pareto_sweep(
        tau_conv=session["tau_conv"],
        tau_fatigue=session["tau_fat"],
        mu_conv=session["mu_conv"],
        mu_opt=session["mu_opt"],
        greedy_prob=session["greedy_prob"],
        logged=test["action"].to_numpy(int),
        y_conv=test["y_conversion"].to_numpy(np.float64),
        y_opt=test["y_optout"].to_numpy(np.float64),
        recency=test["recency"].to_numpy(np.float64),
        propensity=session["propensity"],
        channel_cap=channel_cap,
        budgets=budgets,
        seed=session["seed"],
    )


def _exact_closed_loop(session: dict, channel_cap: float, budget: float) -> dict:
    test = session["test"]
    point = closed_loop_marker(
        tau_conv=session["tau_conv"],
        tau_fatigue=session["tau_fat"],
        mu_conv=session["mu_conv"],
        mu_opt=session["mu_opt"],
        logged=test["action"].to_numpy(int),
        y_conv=test["y_conversion"].to_numpy(np.float64),
        y_opt=test["y_optout"].to_numpy(np.float64),
        propensity=session["propensity"],
        channel_cap=channel_cap,
        budget=budget,
    )
    point["channel_cap"] = float(channel_cap)
    return point


def _pareto_chart(cap: float, title: str, sweep: dict) -> dict:
    loop = _levels(
        [
            {
                "name": "Closed-loop",
                "opt_out": row["opt_out"],
                "inc_conv": row["inc_conv"],
                "budget": row["budget"],
                "channel_cap": cap,
            }
            for row in sweep["closed_loop"]
        ],
        "budget",
    )
    series = [{"name": "Closed-loop", "points": loop}]
    for name in PARETO_POLICY_ORDER:
        row = next(item for item in sweep["policies"] if item["name"] == name)
        series.append(
            {
                "name": name,
                "points": [
                    {
                        "name": name,
                        "opt_out": row["opt_out"],
                        "inc_conv": row["inc_conv"],
                        "channel_cap": cap,
                        "level": 1.0,
                    }
                ],
            }
        )
    return {
        "title": title,
        "kind": "opt_out",
        "channel_budget": cap,
        "series": series,
    }


def _pareto_figures(session: dict) -> list[dict]:
    """Two opt-out sweeps at a fixed channel cap. Caps are 0.5%, 1%, 2.5%, and 5%."""
    titles = (
        (0.15, "Tight channel cap 15%, opt-out cap range 0.5–5%"),
        (0.30, "Relaxed channel cap 30%, opt-out cap range 0.5–5%"),
    )
    figures = []
    for cap, title in titles:
        figure = _pareto_chart(cap, title, _pareto_kwargs(session, cap, PARETO_OPT_BUDGETS))
        if abs(cap - 0.30) < 1e-9:
            figure["marked"] = _exact_closed_loop(session, cap, 0.015)
        figures.append(figure)
    return figures


def _pareto_horizon_figures(session: dict, *, weekly_limit: bool) -> list[dict]:
    """Two channel caps, each point a 21-morning average. The weekly limit can be turned off."""
    test = session["test"]
    if weekly_limit:
        f_grid = session["f_grid"]
        tau_conv_grid = session["tau_conv_grid"]
        tau_fat_grid = session["tau_fat_grid"]
        weekly_cap = WEEKLY_TOUCHES
        seed = session["seed"] + 90
    else:
        # A push every morning settles near tiredness 5.2. The stored grid stops at 4.
        f_grid = np.linspace(0.0, 6.0, 25)
        tau_conv_grid, tau_fat_grid = session["uplift"].predict_tau_grid(session["x_test"], f_grid)
        weekly_cap = PREPARED_HORIZON + 1
        seed = session["seed"] + 190
    greedy_grid = session["uplift"].predict_greedy_grid(session["x_test"], f_grid)
    titles = (
        (0.15, "Tight channel cap 15%, 21-day mean, opt-out cap range 0.5–5%"),
        (0.30, "Relaxed channel cap 30%, 21-day mean, opt-out cap range 0.5–5%"),
    )
    figures = []
    for cap, title in titles:
        sweep = horizon_pareto(
            baseline=test["baseline"].to_numpy(np.float64),
            type_code=test["type_code"].to_numpy(int),
            recency=test["recency"].to_numpy(np.float64),
            tau_conv_grid=tau_conv_grid,
            tau_fat_grid=tau_fat_grid,
            greedy_grid=greedy_grid,
            f_grid=f_grid,
            channel_cap=cap,
            budgets=PARETO_OPT_BUDGETS,
            traffic_shift=PREPARED_SHIFT,
            horizon=PREPARED_HORIZON,
            seed=seed,
            weekly_cap=weekly_cap,
        )
        figures.append(_pareto_chart(cap, title, sweep))
    return figures


def _horizon_side(sim: dict) -> dict:
    summary = sim["summary"]
    return {
        "mean_inc_conv": summary["mean_inc_conv_closed"],
        "mean_opt_out": summary["mean_opt_out_closed"],
        "channel_mix": sim["channel_mix"],
    }


def _distillation_tally(session: dict, cases: list[dict]) -> dict:
    """21-day tally of the standard rule and of the student, at the caps the student copied."""
    budget, cap = 0.015, 0.30
    teacher = next(
        item["simulate"]
        for item in cases
        if abs(item["opt_out_budget"] - budget) < 1e-9 and abs(item["channel_budget"] - cap) < 1e-9
    )
    student = _simulate(session, budget, cap, PREPARED_SHIFT, PREPARED_HORIZON, predictor="student")
    return {
        "opt_out_budget": budget,
        "channel_budget": cap,
        "traffic_shift": PREPARED_SHIFT,
        "horizon_days": PREPARED_HORIZON,
        "teacher": _horizon_side(teacher),
        "student": _horizon_side(student),
    }


def prepared_cases(session: dict) -> dict:
    cached = session.get("prepared")
    if cached is not None:
        return cached
    cases = []
    for title, budget, cap in PREPARED_CASES:
        cases.append(
            {
                "title": title,
                "opt_out_budget": budget,
                "channel_budget": cap,
                "traffic_shift": PREPARED_SHIFT,
                "horizon_days": PREPARED_HORIZON,
                "simulate": _simulate(session, budget, cap, PREPARED_SHIFT, PREPARED_HORIZON),
                "labels": _labels(session, budget, cap),
            }
        )
    payload = {
        "source": session["source"],
        "note": session["note"],
        "traffic_shift": PREPARED_SHIFT,
        "horizon_days": PREPARED_HORIZON,
        "qini_sim": session["qini_sim"],
        "qini_hillstrom": session["qini_hillstrom"],
        "pareto_figures": _pareto_figures(session),
        "pareto_horizon_figures": _pareto_horizon_figures(session, weekly_limit=True),
        "pareto_horizon_open_figures": _pareto_horizon_figures(session, weekly_limit=False),
        "cases": cases,
        "distillation": _distillation_tally(session, cases),
    }
    session["prepared"] = payload
    return payload


def get_session() -> dict:
    global _session
    with _lock:
        if _session is None:
            _session = build_session()
        return _session


def _archetype_rows(test, tau_conv: np.ndarray, actions: np.ndarray) -> list[int]:
    """One user of each story type, preferring the channel that type is built to illustrate."""
    codes = test["type_code"].to_numpy(int)
    preferred = {
        TYPE_PERSUADABLE: PUSH,
        TYPE_SURE: AUDIO,
        TYPE_SLEEPING: CONTROL,
    }
    picked = []
    for code in (TYPE_PERSUADABLE, TYPE_SURE, TYPE_SLEEPING):
        matches = np.flatnonzero(codes == code)
        if len(matches) == 0:
            continue
        pool = matches[actions[matches] == preferred[code]]
        if len(pool) == 0:
            pool = matches
        center = tau_conv[matches].mean(axis=0)
        distance = np.linalg.norm(tau_conv[pool] - center, axis=1)
        picked.append(int(pool[int(np.argmin(distance))]))
    return picked


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/formulation.md")
def formulation_note():
    return FileResponse(FORMULATION, media_type="text/plain; charset=utf-8")


@app.post("/api/simulate")
def simulate(request: SimRequest):
    session = get_session()
    return _simulate(
        session,
        _clip(request.opt_out_budget, 0.002, 0.15),
        _clip(request.channel_budget, 0.05, 0.45),
        _clip(request.traffic_shift, 1.0, 4.0),
        int(np.clip(request.horizon_days, 7, 30)),
        request.predictor,
    )


@app.post("/api/pareto")
def pareto(request: ParetoRequest):
    session = get_session()
    mark = None if request.opt_out_budget is None else _clip(request.opt_out_budget, 0.002, 0.15)
    return _pareto(session, _clip(request.channel_budget, 0.05, 0.45), mark, request.predictor)


@app.post("/api/labels")
def labels(request: InspectRequest):
    session = get_session()
    return _labels(
        session,
        _clip(request.opt_out_budget, 0.002, 0.15),
        _clip(request.channel_budget, 0.05, 0.45),
        request.predictor,
    )


def _message_rates(actions: np.ndarray, outcome: np.ndarray, n_actions: int) -> np.ndarray:
    """Share of yes answers on the training rows, one share per stored message."""
    rates = np.zeros(n_actions, dtype=np.float64)
    actions = np.asarray(actions, dtype=int)
    outcome = np.asarray(outcome, dtype=np.float64)
    for arm in range(n_actions):
        mask = actions == arm
        if mask.any():
            rates[arm] = float(outcome[mask].mean())
    return rates


def _score_chances(chances: dict, answers: dict, truth: dict | None = None) -> dict:
    """One report per split and per answer. ``chances[split][answer]`` is a chance per row."""
    return {
        split: {
            key: chance_report(
                chances[split][key],
                answers[split][key],
                None if truth is None else truth[split][key],
            )
            for key in answers[split]
        }
        for split in answers
    }


def _outcome_grid(frame) -> dict:
    """Buy chance and stop chance for every message, from the simulator formulas."""
    n = len(frame)
    baseline = frame["baseline"].to_numpy(np.float64)
    type_code = frame["type_code"].to_numpy(int)
    fatigue = frame["f"].to_numpy(np.float64)
    arms = [np.full(n, arm, dtype=int) for arm in range(N_ACTIONS)]
    return {
        "buy": np.column_stack([conversion_prob(baseline, type_code, arm, fatigue) for arm in arms]),
        "stop": np.column_stack([optout_prob(arm, fatigue, shift=1.0) for arm in arms]),
    }


def _uplift_block(predicted: dict, truth: dict) -> dict:
    """Predicted grids against the formula grid. ``predicted[split][answer]`` is (n, messages)."""
    return {
        split: {key: uplift_report(predicted[split][key], truth[split][key]) for key in ("buy", "stop")}
        for split in predicted
    }


def _logged_chances(uplift: UpliftModels, features: np.ndarray, actions: np.ndarray) -> dict:
    """Predicted buy chance and stop chance for the message stored on each row."""
    mu_conv, mu_opt = uplift.predict_mu(features)
    rows = np.arange(len(actions))
    chosen = np.asarray(actions, dtype=int)
    return {"buy": mu_conv[rows, chosen], "stop": mu_opt[rows, chosen]}


def _predictor_report(session: dict) -> dict:
    """Train and test fit of each method's chance, for the message stored on the row."""
    frames = {"train": session["train"], "test": session["test"]}
    features = {"train": session["x_train"], "test": session["x_test"]}
    actions = {split: frame["action"].to_numpy(int) for split, frame in frames.items()}
    answers = {
        split: {
            "buy": frame["y_conversion"].to_numpy(np.float64),
            "stop": frame["y_optout"].to_numpy(np.float64),
        }
        for split, frame in frames.items()
    }
    # The chance that drew each stored answer, from the simulator's own formulas.
    truth = {
        split: {
            "buy": conversion_prob(
                frame["baseline"].to_numpy(np.float64),
                frame["type_code"].to_numpy(int),
                actions[split],
                frame["f"].to_numpy(np.float64),
            ),
            "stop": optout_prob(actions[split], frame["f"].to_numpy(np.float64), shift=1.0),
        }
        for split, frame in frames.items()
    }
    grid = {split: _outcome_grid(frame) for split, frame in frames.items()}
    simulator = {}
    for kind in ("tree", "svm", "mlp"):
        uplift = session["uplift"] if kind == "tree" else UpliftModels.fit(
            session["x_train"],
            actions["train"],
            answers["train"]["buy"],
            answers["train"]["stop"],
            seed=session["seed"],
            kind=kind,
        )
        chances = {split: _logged_chances(uplift, features[split], actions[split]) for split in frames}
        predicted = {}
        for split in frames:
            mu_conv, mu_opt = uplift.predict_mu(features[split])
            predicted[split] = {"buy": mu_conv, "stop": mu_opt}
        simulator[kind] = _score_chances(chances, answers, truth)
        simulator[kind]["uplift"] = _uplift_block(predicted, grid)
    rates = {key: _message_rates(actions["train"], answers["train"][key], N_ACTIONS) for key in ("buy", "stop")}
    simulator["message"] = _score_chances(
        {split: {key: rates[key][actions[split]] for key in rates} for split in frames},
        answers,
        truth,
    )
    message_grid = {
        split: {
            key: np.broadcast_to(rates[key], (len(actions[split]), N_ACTIONS)).copy()
            for key in ("buy", "stop")
        }
        for split in frames
    }
    simulator["message"]["uplift"] = _uplift_block(message_grid, grid)
    simulator["coin"] = _score_chances(
        {split: {key: np.full(len(actions[split]), 0.5) for key in ("buy", "stop")} for split in frames},
        answers,
        truth,
    )
    simulator["formula"] = _score_chances(truth, answers, truth)
    simulator["formula"]["uplift"] = _uplift_block(grid, grid)
    return {"simulator": simulator}


@app.get("/api/predictors")
def predictors():
    session = get_session()
    cached = session.get("predictor_report")
    if cached is None:
        cached = _predictor_report(session)
        session["predictor_report"] = cached
    return cached


@app.get("/api/cases")
def cases():
    return prepared_cases(get_session())


@app.post("/api/distill")
def distill(request: DistillRequest):
    session = get_session()
    report = distill_policy(
        session["uplift"],
        session["x_train"],
        session["x_test"],
        budget=_clip(request.opt_out_budget, 0.002, 0.15),
        channel_cap=_clip(request.channel_budget, 0.05, 0.45),
        epochs=request.epochs,
        seed=session["seed"],
    )
    report["source"] = session["source"]
    return report


@app.post("/api/user-inspect")
def user_inspect(request: InspectRequest):
    session = get_session()
    test = session["test"]
    n = len(test)
    budget = _clip(request.opt_out_budget, 0.002, 0.15)
    cap = _clip(request.channel_budget, 0.05, 0.45)
    eligible = eligibility(np.zeros(n))
    actions, lam, mu, scores = solve_assignment(
        session["tau_conv"],
        session["tau_fat"],
        budget=budget,
        channel_cap=cap,
        eligible=eligible,
    )
    greedy = assign_ranked(session["greedy_prob"], cap, eligible, allow_nonpositive=True)
    users = []
    for row in _archetype_rows(test, session["tau_conv"], actions):
        record = test.iloc[row]
        channels = []
        for arm in range(N_ACTIONS):
            channels.append(
                {
                    "name": ACTIONS[arm],
                    "tau_conv": float(session["tau_conv"][row, arm]),
                    "tau_fatigue": float(session["tau_fat"][row, arm]),
                    "score": float(scores[row, arm]),
                    "chosen": bool(int(actions[row]) == arm),
                }
            )
        chosen = int(actions[row])
        card = {
            "type": TYPE_NAMES[int(record["type_code"])],
            "action": ACTIONS[chosen],
            "greedy_action": ACTIONS[int(greedy[row])],
            "fatigue": float(record["f"]),
            "features": {
                "recency": float(record["recency"]),
                "history": float(record["history"]),
                "mens": int(record["mens"]),
                "womens": int(record["womens"]),
                "newbie": int(record["newbie"]),
                "zip_code": str(record["zip_code"]),
                "channel": str(record["channel"]),
            },
            "channels": channels,
        }
        if chosen == AUDIO:
            card["audio"] = bitrate_illustration()
        users.append(card)
    return {
        "source": session["source"],
        "note": session["note"],
        "lambda": float(lam),
        "mu": [float(value) for value in mu],
        "users": users,
    }
