"""Unit checks for the decision math and a smoke test of the four routes."""

import os

os.environ["ORCH_FALLBACK"] = "1"
os.environ["ORCH_N"] = "400"
os.environ["ORCH_SEED"] = "0"

import numpy as np
import pandas as pd

from app.data import (
    BETA,
    CONTROL,
    GAMMA,
    N_ACTIONS,
    PUSH,
    assign_types,
    fatigue_step,
    feature_matrix,
    generate_fallback,
    make_cross_section,
    prepare_demo,
    split_counts,
)
from app.models import fit_dr_cate, predict_dr_tau
from app.ope import dr_value, ips_value
from app.optimizer import solve_assignment


def test_split_counts_match_the_email_file_and_a_small_cohort():
    assert split_counts(64000) == (48000, 8000, 8000)
    demo = prepare_demo(generate_fallback(400, seed=0), n=400, seed=0)
    cross = make_cross_section(demo, seed=0)
    counts = cross["split"].value_counts().sort_index()
    assert counts.tolist() == [300, 50, 50]


def test_fallback_is_deterministic():
    first = generate_fallback(120, seed=7)
    second = generate_fallback(120, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_types_cover_four_regions():
    frame = generate_fallback(2000, seed=0)
    codes = assign_types(
        frame["history"].to_numpy(float),
        frame["recency"].to_numpy(float),
        frame["newbie"].to_numpy(int),
    )
    assert set(np.unique(codes).tolist()) == {0, 1, 2, 3}


def test_fatigue_mean_step_matches_recursion():
    pushed = fatigue_step(np.array([1.0]), np.array([PUSH]), eps=0.0)
    assert pushed[0] == np.float64(GAMMA * 1.0 + BETA[PUSH])
    held = fatigue_step(np.array([1.0]), np.array([CONTROL]), eps=0.0)
    assert held[0] == np.float64(GAMMA * 1.0)


def test_control_score_is_zero_and_wins_when_actions_hurt():
    tau_conv = np.array([[0.0, -0.2, -0.4, -0.1]])
    tau_fat = np.array([[0.0, 0.2, 0.3, 0.1]])
    actions, _, _, scores = solve_assignment(
        tau_conv,
        tau_fat,
        budget=1.0,
        channel_cap=1.0,
        eligible=np.ones((1, N_ACTIONS), dtype=bool),
    )
    assert actions[0] == CONTROL
    assert scores[0, CONTROL] == 0.0


def test_lambda_rises_when_over_budget_and_falls_when_under():
    n = 120
    tau_conv = np.zeros((n, N_ACTIONS))
    tau_fat = np.zeros((n, N_ACTIONS))
    tau_conv[:, PUSH] = np.linspace(0.02, 0.6, n)
    tau_fat[:, PUSH] = 0.05
    eligible = np.ones((n, N_ACTIONS), dtype=bool)
    tight, lam_tight, _, _ = solve_assignment(
        tau_conv, tau_fat, budget=0.02, channel_cap=1.0, eligible=eligible
    )
    predicted = float(tau_fat[np.arange(n), tight].mean())
    assert lam_tight > 0.0
    assert predicted <= 0.02 + 1e-6

    loose, lam_loose, _, _ = solve_assignment(
        tau_conv, tau_fat, budget=0.5, channel_cap=1.0, eligible=eligible
    )
    assert lam_loose == 0.0
    assert np.all(loose == PUSH)

    _, lam_from_high, _, _ = solve_assignment(
        tau_conv,
        tau_fat,
        budget=0.02,
        channel_cap=1.0,
        eligible=eligible,
        lambda_init=50.0,
        n_steps=5,
    )
    assert lam_from_high < 50.0


def test_weekly_cap_forces_control():
    n = 8
    tau_conv = np.ones((n, N_ACTIONS))
    tau_fat = np.zeros((n, N_ACTIONS))
    eligible = np.ones((n, N_ACTIONS), dtype=bool)
    eligible[:, 1:] = False
    actions, _, _, _ = solve_assignment(
        tau_conv, tau_fat, budget=1.0, channel_cap=1.0, eligible=eligible
    )
    assert np.all(actions == CONTROL)


def test_dr_and_ips_are_finite_under_uniform_logging():
    rng = np.random.default_rng(0)
    n = 60
    mu = rng.random((n, N_ACTIONS))
    logged = rng.integers(0, N_ACTIONS, n)
    policy = rng.integers(0, N_ACTIONS, n)
    rewards = rng.random(n)
    propensity = np.full(N_ACTIONS, 0.25)
    assert np.isfinite(dr_value(mu, logged, policy, rewards, propensity))
    assert np.isfinite(ips_value(logged, policy, rewards, propensity))


def test_dr_cate_control_contrast_is_zero():
    demo = prepare_demo(generate_fallback(360, seed=2), n=360, seed=2)
    cross = make_cross_section(demo, seed=2)
    features = feature_matrix(cross, cross["f"].to_numpy(float))
    models = fit_dr_cate(
        features,
        cross["action"].to_numpy(int),
        cross["y_conversion"].to_numpy(float),
        np.full(N_ACTIONS, 0.25),
        seed=2,
    )
    tau = predict_dr_tau(models, features[:24])
    assert tau.shape == (24, N_ACTIONS)
    assert np.allclose(tau[:, CONTROL], 0.0)
    assert np.isfinite(tau).all()


def test_api_routes():
    from fastapi.testclient import TestClient

    from app.main import app, reset_state

    reset_state()
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Closed-loop" in page.text

    simulated = client.post("/api/simulate", json={"horizon_days": 8})
    assert simulated.status_code == 200
    body = simulated.json()
    assert body["source"] == "fallback"
    assert len(body["days"]) == 8
    assert len(body["lambda"]) == 8

    frontier = client.post("/api/pareto", json={})
    assert frontier.status_code == 200
    payload = frontier.json()
    assert len(payload["closed_loop"]) >= 2
    assert len(payload["policies"]) == 4
    assert len(payload["qini_sim"]["fraction"]) >= 2
    assert len(payload["qini_hillstrom"]["fraction"]) >= 2

    student = client.post("/api/distill", json={"epochs": 2})
    assert student.status_code == 200
    fidelity = student.json()["fidelity"]
    assert 0.0 <= fidelity <= 1.0

    inspected = client.post("/api/user-inspect", json={})
    assert inspected.status_code == 200
    users = inspected.json()["users"]
    assert len(users) == 3
    assert len({user["type"] for user in users}) == 3

    labels = client.post("/api/labels", json={"opt_out_budget": 0.015, "channel_budget": 0.15})
    assert labels.status_code == 200
    label_body = labels.json()
    assert label_body["rows"]
    opened = label_body["rows"][0]["open"]
    capped = label_body["rows"][0]["capped"]
    for block in (opened, capped):
        for key in ("unconstrained", "charged", "greedy"):
            assert abs(sum(block[key].values()) - 1.0) < 1e-6
        assert 0.0 <= block["match_charged_unconstrained"] <= 1.0
        assert 0.0 <= block["match_charged_greedy"] <= 1.0

    prepared = client.get("/api/cases")
    assert prepared.status_code == 200
    body = prepared.json()
    cases = body["cases"]
    assert len(cases) == 4
    assert cases[0]["channel_budget"] == 0.15
    figures = body["pareto_figures"]
    assert [figure["kind"] for figure in figures] == ["opt_out", "opt_out"]
    assert [round(point["budget"], 4) for point in figures[0]["series"][0]["points"]] == [0.005, 0.01, 0.025, 0.05]
    assert figures[0]["channel_budget"] == 0.15
    assert figures[1]["channel_budget"] == 0.30
    assert "0.5" in figures[0]["title"] and "5%" in figures[1]["title"]
    horizon = body["pareto_horizon_figures"]
    assert [figure["kind"] for figure in horizon] == ["opt_out", "opt_out"]
    assert [round(point["budget"], 4) for point in horizon[0]["series"][0]["points"]] == [0.005, 0.01, 0.025, 0.05]
    assert horizon[0]["channel_budget"] == 0.15 and horizon[1]["channel_budget"] == 0.30
    assert len(horizon[0]["series"]) == 5 and len(horizon[0]["series"][1]["points"]) == 1
    opened = body["pareto_horizon_open_figures"]
    assert [figure["kind"] for figure in opened] == ["opt_out", "opt_out"]
    assert [round(point["budget"], 4) for point in opened[0]["series"][0]["points"]] == [0.005, 0.01, 0.025, 0.05]
    assert opened[0]["channel_budget"] == 0.15 and opened[1]["channel_budget"] == 0.30
    assert len(opened[0]["series"]) == 5
    distill = body["distillation"]
    assert distill["opt_out_budget"] == 0.015
    assert distill["channel_budget"] == 0.30
    assert distill["horizon_days"] == 21
    for side in ("teacher", "student"):
        assert 0.0 <= body["distillation"][side]["mean_opt_out"] <= 1.0
        assert abs(sum(body["distillation"][side]["channel_mix"].values()) - 1.0) < 1e-6
    assert "Simulate custom" in page.text
