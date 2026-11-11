# Closed-loop message orchestrator

A shop, a music app, or a delivery app can send one note: a buzz, an email, a short clip inside the app, or silence. The note can bring a sale. The same note can make a person say stop. This project chooses, each morning, who should hear something and who should be left alone.

The quantity that matters is the extra the note causes, compared with silence for the same person. A person who would have bought anyway is not a win for the note. The morning also stays inside two limits: how many extra stops the notes may cause, and how many people may hear one channel. Tomorrow is not a copy of today. A note leaves the person more tired, and the charges that enforce the limits start the next morning from the values they finished with.

The same choice shows up for a song or a post. Once some other model has scored it, the decision is still one action under a budget: distribute it, show it less, hold it for review, or leave it alone. This repository fills that action in as a message and the cost in as notification fatigue. It does not train a content encoder, a moderation classifier, or a catalog-enforcement model.

A greedy rule sends the note with the highest raw buy chance. That often spends a loud note on someone who would have bought anyway. The closed loop sends the note where it adds a buy, charges the extra stops, and moves that charge again when the population's tiredness changes.

The formulas, in the same shop language as the page, are in [`docs/mathematical_formulation.md`](docs/mathematical_formulation.md). The build targets live in [`multi_objective_decisioning_messaging_orchestrator_project_report_build_spec.md`](multi_objective_decisioning_messaging_orchestrator_project_report_build_spec.md). The dashboard prints measured values. Those targets are not pasted in as results.

## What one morning does

1. Read a shopping card and today's tiredness.
2. Ask a separate predictor, for each message, for a buy rate and a stop rate.
3. Score a message by its extra buy over silence, minus a charge for the extra stops and a charge for a channel that is near its share.
4. Send the message with the best score. Silence wins when no note clears the charges.
5. Roll the day's buy and stop from the formulas, with one shared draw against silence, so one person contributes +1, 0, or −1.
6. Carry tiredness forward. A push adds more than an email, an email more than audio, and a quiet day lets tiredness fade. At most 3 notes in 7 days.
7. On Saturday and Sunday the stop formula is busier. The charge sees predicted extra stops multiplied by 1.8 before it moves. The buy formula does not change.

The prepared charts use histogram gradient-boosted trees: 40 trees, depth 3, at least 50 people in a leaf, learning rate 0.08, squared error, no early stopping. Columns stay in their own units. A support-vector machine and a small neural net are other predictors. They rescale the columns. They are scored on the Predictors tab and can be selected under Simulate custom. They are not the source of the prepared charts.

## Repository

| Path | Role |
| --- | --- |
| `app/data.py` | Shopping cards, the response formulas, and the training rows. |
| `app/models.py` | Predictors: trees, SVM, neural net, doubly robust contrast, greedy fit, and the student copy. |
| `app/optimizer.py` | The charge search, the channel and weekly caps, and the 21-morning loop. |
| `app/ope.py` | Qini curves, the one-morning filled-in value, and the Pareto sweeps. |
| `app/main.py` | FastAPI app, the saved-result reader, and the live Simulate custom routes. |
| `static/index.html` | The whole dashboard: overview, formulation, benchmarks, and results. |
| `docs/mathematical_formulation.md` | The formulation the page serves at `/formulation.md`. |
| `data/hillstrom.csv` | The public Hillstrom shopping cards, about 64,000 people. |
| `prepared/cases.json` | Saved Qini, Pareto, shadow-price, distillation, and segment results. |
| `prepared/predictors.json` | Saved predictor scores for the fixed Predictors tab. |
| `experiments/` | Offline comparisons. They do not feed the dashboard. |
| `tests/test_pipeline.py` | Formula checks and a small end-to-end run of the API. |

## Python

### `app/data.py`

The Hillstrom file supplies the card: recency, history, mens, womens, zip code, newbie, and the usual shopping channel. It does not record tiredness, a push, or a clip inside the app. `load_hillstrom` reads `data/hillstrom.csv` and downloads the public file into that path if it is missing. `generate_fallback` builds a seeded table with the same columns when the file cannot be read.

`prepare_demo` and `make_cross_section` turn each card into one training row. Tiredness is drawn uniformly from 0 to 3. The message is drawn uniformly from silence, email, push, and audio. A written buy formula and a written stop formula produce two chances. A coin turns each chance into 1 or 0. The trees study those coins. The chances stay off the row.

Of about 64,000 people, 48,000 fit the predictors, 8,000 choose the settings and then join that fit, and 8,000 stay held out. `ORCH_N` shrinks that cohort. The default `0` keeps the full file.

The buy chance is a sigmoid of a card score plus a segment-and-message lift, faded by `exp(-0.65 * tiredness)`. The four segments are Sure Thing, Persuadable, Lost Cause, and Sleeping Dog. The stop chance depends on tiredness, a typed message cost, and a weekend term. Training rows use a weekday, so that term is 0. A later weekend adds `log 1.8`. Fatigue fades by 0.85 each day and then jumps: 0.78 for a push, 0.40 for email, 0.20 for audio, and 0 for silence.

### `app/models.py`

`UpliftModels.fit` is a T-learner. It fits eight models, one for each message and each answer, plus one more model of the raw buy chance. That last model is the greedy predictor. It sees the message as four columns and does not subtract silence.

`make_hgb` builds the histogram gradient-boosted trees used on the prepared charts. `kind="svm"` and `kind="mlp"` build the other predictors. Those two scale the columns. The trees do not. Predictions are clipped to the range 0 to 1. The extra for a message is its predicted rate minus the silence rate.

`fit_dr_cate` and `predict_dr_tau` build a second, doubly robust estimate of that extra. The live prepared policy uses the plain subtraction. The doubly robust estimate is what Simulate custom uses when it fills in a person who was not sent the message the rule wanted. Tests check the doubly robust path.

`fit_student` trains a smaller `HistGradientBoostingClassifier` to name the message the charged rule assigned. It does not search for a new charge. The channel cap and the weekly skip are applied to the name it returns. The prepared distillation copies the rule at a 1.5% opt-out cap and a 30% channel cap.

### `app/optimizer.py`

`solve_assignment` runs 80 passes. Each pass gives every eligible person the message with the best score

```text
extra buy − λ × extra stop − μ_message
```

Silence scores 0. λ rises when the average predicted extra stop is over the cap and falls when it is under. It does not go below 0. Each channel charge μ moves the same way against that channel's share. A repair then turns the weakest notes into silence if the predicted extra stops are still over the cap.

`simulate_horizon` plays 21 mornings. Tiredness starts at 0. The closed-loop rule starts each morning from yesterday's charges. On a weekend it multiplies predicted extra stops by the traffic shift before the charge moves. The frozen rule keeps Monday's opt-out charge and plans the day with the weekday predictions. Both rules carry their own tiredness. The recorded extra buy and extra stop are the mean of the shared-draw gaps, then the mean of the 21 days.

`horizon_pareto` repeats that loop for the closed-loop rule at several opt-out caps, and for random, a static heuristic, greedy, and unconstrained uplift. The weekly limit can be turned off by raising the cap above the horizon.

### `app/ope.py`

`qini_curve` sorts the held-out people by the largest extra buy among email, push, and audio. Silence is not a candidate. A stored message that matches the chosen note and a buy counts as +4. A buy under silence counts as −4. Each message was stored for about one person in four, which is why the credit is 4. The height divides by everyone on the list, including people not yet reached.

`pareto_sweep` is the one-morning chart on Simulate custom. It is separate from the 21-day Pareto. A person who was not sent the message the rule wanted is filled in from the tree chance. When the stored message matches, the coin corrects that chance, and a message sent to one person in four is counted four times.

### `app/main.py`

FastAPI serves the page and the formulation. The fixed Results tabs read JSON when the process is the full run: `ORCH_FALLBACK` is not `1`, `ORCH_N` is 0, and `ORCH_SEED` is 0.

| Route | What it returns |
| --- | --- |
| `GET /` | `static/index.html` |
| `GET /formulation.md` | The formulation, read from disk on each request. |
| `GET /api/cases` | Saved `prepared/cases.json`, or a fresh 21-day run if that file is absent or the cohort is not the full one. |
| `GET /api/predictors` | Saved `prepared/predictors.json`, with the same fallback. |
| `POST /api/simulate` | A live horizon at the requested caps, shift, horizon, and predictor. |
| `POST /api/pareto` | The live one-morning Pareto for Simulate custom. |
| `POST /api/labels` | Live segment mixes for one morning. |
| `POST /api/distill` | Agreement between the student and the teacher at the requested caps. |
| `POST /api/user-inspect` | Three held-out people and the rates the predictors give them. |

Simulate custom is the only Results tab that calls the live routes. The first time it runs, the server fits the trees. Changing the predictor to SVM, the neural net, or the student fits that copy once and then reuses it.

## Saved results and experiments

`prepared/cases.json` and `prepared/predictors.json` are the full Hillstrom run at seed 0. Refreshing the fixed tabs reads those files. They are not recomputed on each page load.

`experiments/compare_predictors.py` compares tree, SVM, and neural-net settings by squared error and writes `experiments/predictor_fit.json`. `experiments/compare_logloss.py` is a separate log-loss comparison and writes `experiments/predictor_logloss.json`, `predictor_logloss_newbuy.json`, and `predictor_logloss_buy70.json`. The dashboard does not load these files. The live loss remains squared error.

## Run

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. With the prepared files present, the fixed charts appear from disk. Simulate custom still fits on first use.

The same fixed page is also at `https://ag-68.github.io/Causal_Decisioning_Multi_Objective_Orchestrator/`. That copy reads `docs/prepared/` and has no live simulator. Its Simulate custom tab links here.

```bash
ORCH_FALLBACK=1 ORCH_N=400 ORCH_SEED=0 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`ORCH_FALLBACK=1` skips the Hillstrom file and uses the seeded stand-in. The page says so. `ORCH_N` is the cohort size. `ORCH_SEED` fixes the draws. A non-default cohort does not read `prepared/`.

```bash
pytest
```

The tests set `ORCH_FALLBACK=1`, `ORCH_N=400`, and `ORCH_SEED=0`, then check the split, the fatigue step, the charge search, the doubly robust value, and the API, including that `qini_hillstrom` is still returned.

The dashboard is one dark page. Formulation, benchmarks, and results are tabs in `static/index.html`. Plotly draws the charts. KaTeX draws the formulas.

```bash
docker build -t message-orchestrator .
docker run --rm -p 8080:8080 message-orchestrator
```

The image listens on port 8080. It copies `app/` and `static/`. It is not deployed from this repository.
