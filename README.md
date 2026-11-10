# Closed-loop causal message orchestrator

CPU demo of constrained causal decisioning: which channel, if any, reaches which user.

The executive spec is `multi_objective_decisioning_messaging_orchestrator_project_report_build_spec.md`. The formulas are in `docs/mathematical_formulation.md`.

## Shopping cards and the simulator

- **Shopping cards.** The public Hillstrom file supplies recency, history, mens, womens, zip code, newbie, and channel for about 64,000 people. The loader caches it at `data/hillstrom.csv`. If the download fails, a seeded table with the same columns is used and the UI says so.
- **Simulator.** Those cards feed potential outcomes for Control, Email, Push, and In-App Audio, plus a fatigue state. Opt-out, the Pareto curve, the shadow-price path, the Qini curve, and the five-policy comparison live here. Those response parameters are scenario settings. They are not fit from the shopping cards.

Design targets in the executive spec (lift retained, opt-out reduction, volume, fidelity, latency) are targets. The dashboard prints measured values.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The first request fits the boosters.

`ORCH_FALLBACK=1` skips the download. `ORCH_N` is the decision-cohort size. The default `0` keeps the full file, about 64,000 people: 48,000 fit the predictors, 8,000 choose their settings and then join the fit, and 8,000 stay held out. `ORCH_SEED` fixes the simulator.

```bash
pytest
```

The image listens on port 8080. It is not deployed from this repository.

```bash
docker build -t message-orchestrator .
docker run --rm -p 8080:8080 message-orchestrator
```
