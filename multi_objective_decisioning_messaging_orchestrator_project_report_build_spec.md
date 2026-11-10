# Closed-Loop Causal Decisioning & Multi-Objective Message Orchestrator
**Executive Project Report & Technical Specification**  
**Author:** Ahmet Hasim Gökceoglu, Ph.D. | **Portfolio:** [ag-68.github.io](https://ag-68.github.io/)

---

## 1. Executive Summary & Plain-English Problem Statement

Consumer platforms operating at scale (such as Spotify’s Messaging & Personalization teams, King’s Decision Automation platform, and fintech/marketplace growth engines) face a fundamental conflict every time they communicate with a user: **how to decide *which* message or experience reaches *which* user, through *which* channel (Push, Email, In-App Audio/Voice Teaser, or Silent/No-Op), and at *what* moment.**

Standard industrial machine learning pipelines fall into three common traps when solving this problem:
1. **Confusing Correlation with True Impact (Wasted Nudges & "Sleeping Dogs"):** Standard predictive ML models estimate *who is likely to convert*, not *whether sending a message actually changes their behavior*. As a result, greedy models spam loyal "Sure Thing" users who would have converted anyway, while accidentally triggering "Sleeping Dogs"—frustrated users who unsubscribe or churn specifically because an intrusive pop-up or notification interrupted them.
2. **Brittle, Hand-Tuned Multi-Objective Weights:** Sending more notifications increases day-1 clicks but causes notification fatigue, app opt-outs, and long-term churn. Most product teams try to balance these competing goals using a static, hand-tuned formula such as $\text{Score} = P(\text{Convert}) - \alpha \cdot P(\text{Opt-Out})$. Product managers spend months guessing the "magic weight" $\alpha$ via slow A/B tests—and the moment user traffic shifts on a weekend or seasonal campaign, a fixed $\alpha$ breaks, either blowing past safety guardrails or leaving conversions on the table.
3. **Myopic, Siloed Triggers Across Channels:** Separate teams (Push, Email, In-App Promotions) trigger messages independently based on static snapshots of the user, ignoring how sending a push notification *today* spikes the user's annoyance and degrades their receptiveness *tomorrow*.

### The Solution
This project implements a production-grade, CPU-efficient **Closed-Loop Causal & Multi-Objective Orchestration Engine** (paired with an interactive web simulator and offline policy evaluator). Instead of relying on static predictive scores and manual penalty weights, it combines **Causal Uplift Modeling (CATE)** with **Control-Theoretic Primal-Dual Constrained Optimization** and **Dynamic Fatigue State-Space Tracking**—bringing proven mathematical resource-allocation principles from real-time telecommunications R&D into consumer product decisioning.

---

## 2. Core Technical Differentiators (Why This Beats Standard Product ML)

Rather than competing on raw compute scale, this architecture demonstrates three foundational mathematical capabilities rarely found in standard data-science pipelines. The general constrained decision problem, the reading of each symbol, and the uplift, shadow-price, fatigue, and evaluation steps are written out in [docs/mathematical_formulation.md](docs/mathematical_formulation.md).

### Differentiator 1: Auto-Tuning Primal-Dual "Shadow Price" Control (Eliminating Magic Weights)
* **The Standard ML Limitation:** Static linear scalarization ($\text{Reward} - \alpha \cdot \text{Risk}$) cannot guarantee hard operational guardrails when data distributions shift.
* **Our Mathematical Approach:** We formulate multi-channel message routing as a **constrained optimization problem**: maximize net incremental conversions across the user cohort subject to hard upper bounds on global notification opt-out rate ($\le B_{\text{opt-out}}$, e.g., $1.5\%$) and per-channel volume budgets.
  $$\max_{\pi} \sum_{i=1}^{N} \sum_{a \in \mathcal{A}} \pi(a \mid x_i) \, \hat{\tau}_{\text{conv}}(x_i, a) \quad \text{s.t.} \quad \frac{1}{N}\sum_{i=1}^{N} \sum_{a \in \mathcal{A}} \pi(a \mid x_i) \, \hat{\tau}_{\text{fatigue}}(x_i, a) \le B_{\text{opt-out}}$$
* **Closed-Loop Execution:** Using **Lagrangian Primal-Dual Gradient Ascent**, the orchestrator maintains a dynamic **shadow price ($\lambda_t \ge 0$)** representing the real-time "cost of annoying a user." Whenever user fatigue approaches the guardrail $B_{\text{opt-out}}$, $\lambda_t$ automatically rises to throttle back intrusive channels (like Push) and shift traffic toward lower-fatigue channels (like In-App Audio Teasers or Do-Nothing). When slack is available, $\lambda_t$ relaxes—guaranteeing constraint satisfaction without human weight-tuning.

### Differentiator 2: Joint Multi-Channel "Water-Filling" & Fast Policy Distillation
* **The Standard ML Limitation:** Independent per-channel threshold rules cause conflicting treatments (e.g., firing both an email and a push notification on the same day) or cannibalize limited user attention.
* **Our Mathematical Approach:** Inspired by cross-layer multi-user scheduling and OFDM water-filling in wireless systems, the central arbiter evaluates the **net dual-adjusted uplift** across all candidate channels $a \in \{\text{Push}, \text{Email}, \text{In-App Audio Teaser}, \text{Control}\}$ simultaneously:
  $$a_i^* = \arg\max_{a \in \mathcal{A}} \left[ \hat{\tau}_{\text{conv}}(x_i, a) - \lambda_{\text{fatigue}} \cdot \hat{\tau}_{\text{fatigue}}(x_i, a) - \mu_a \right]$$
  where $\mu_a$ is the shadow price for channel $a$'s capacity limit. Furthermore, we demonstrate **Teacher–Student Policy Distillation**, compressing the full causal ensemble + Lagrangian solver into a lightweight student MLP capable of $<1\text{ ms}$ real-time edge/server inference.

### Differentiator 3: Dynamic "Fatigue State-Space" Tracking for Long-Horizon Value
* **The Standard ML Limitation:** Treating user features as static rows ignores temporal dynamics—specifically, how over-messaging today erodes long-term reachability over a 30-day horizon.
* **Our Mathematical Approach:** We model each user's latent notification fatigue $f_{i,t}$ as a **first-order dynamical state-space process** with exponential recovery:
  $$f_{i, t+1} = \gamma \, f_{i, t} + \beta_a \cdot \mathbb{I}(a_{i,t} \neq \text{Control}) + \epsilon_{i,t}, \quad \gamma \in (0, 1)$$
  As $f_{i,t}$ accumulates, the user's probability of permanently disabling notifications spikes non-linearly, while the causal conversion uplift $\tau_{\text{conv}}(x_{i,t}, f_{i,t}, a)$ decays. Accounting for this state transition allows our policy to preserve long-term subscriber reachability over multi-week horizons.

---

## 3. System Architecture & Audio/Voice Personalization Hook

```
[User Context + Dynamic Fatigue State f_{i,t}]
                     │
                     ▼
   ┌───────────────────────────────────┐
   │  Causal Uplift & Risk Estimators  │
   │  • CATE Conversion Lift: τ_conv   │
   │  • CATE Opt-Out Risk:   τ_fatigue │
   └─────────────────┬─────────────────┘
                     │
                     ▼
   ┌───────────────────────────────────┐     ┌──────────────────────────────────┐
   │ Primal-Dual Lagrangian Arbitrator │◄────┤ Live Guardrail Controller        │
   │ • Auto-tunes Shadow Price (λ_t)   │     │ • Max Opt-Out Budget (e.g. 1.5%) │
   │ • Enforces Channel Caps (μ_a)     │     │ • Max Weekly Touches per User    │
   └─────────────────┬─────────────────┘     └──────────────────────────────────┘
                     │
       ┌─────────────┼─────────────┬────────────────────┐
       ▼             ▼             ▼                    ▼
   [Control]      [Email]   [Push Notification]  [In-App Voice/Audio Teaser]
  (No Action)                                    (Neural Codec / Expressive TTS
                                                  Bitrate vs. Latency Preview)
```

* **Multi-Modal / Audio Personalization Hook:** When the orchestrator selects the **In-App Voice/Audio Teaser** channel for high-affinity listeners, the UI previews pre-synthesized expressive audio prompts alongside a **Neural Audio Codec Bitrate vs. Expressiveness Inspector** (illustrating how learned vector quantization codebooks compress continuous audio waveforms at $6\text{ kbps}$ vs. $24\text{ kbps}$).

---

## 4. Benchmarking Methodology, Datasets & Concrete KPIs (100% CPU-Only)

To ensure scientific rigor without requiring GPUs or proprietary company logs, the system is evaluated using a **dual-benchmark protocol** that runs in $<5\text{ seconds}$ on a standard CPU:

### A. Real-World Dataset Benchmark: Hillstrom Multi-Channel Messaging Dataset (64,000 Users)
* **Dataset:** 64,000 real e-commerce/subscription users randomly assigned to *Channel A*, *Channel B*, or *Control (No Message)*, logging historical attributes (`recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`) and true outcomes (`visit`, `conversion`, `spend`).
* **Evaluation Protocol:** We augment the dataset with a transparent, configurable **Fatigue & Opt-Out Response Mechanics layer** (simulating *Persuadables*, *Sure Things*, *Lost Causes*, and *Sleeping Dogs*) and evaluate policies offline using **Doubly Robust (DR) Off-Policy Evaluation (OPE)** and **Area Under the Uplift / Qini Curve (AUUC)**:
  $$\hat{V}_{\text{DR}}(\pi) = \frac{1}{N} \sum_{i=1}^{N} \left[ \hat{\mu}\big(x_i, \pi(x_i)\big) + \frac{\mathbb{I}\big(a_i = \pi(x_i)\big)}{p(a_i \mid x_i)} \Big( r_i - \hat{\mu}(x_i, a_i) \Big) \right]$$

### B. The 5 Benchmarked Policies Compared Side-by-Side
1. **Random / Uniform Baseline:** Assigns channels uniformly up to budget.
2. **Static Heuristic Rule:** Industry rule-of-thumb (e.g., *"Send Push to users inactive $>3$ days, Email otherwise"*).
3. **Greedy Predictive ML (Standard Propensity Classifier):** Trains a standard gradient-boosted/MLP model to predict raw conversion $P(Y=1 \mid X, A)$ and messages the highest scorers (spams *Sure Things* and triggers *Sleeping Dogs*).
4. **Unconstrained Causal Uplift (T-Learner / X-Learner):** Selects the channel with the highest conversion CATE $\hat{\tau}_{\text{conv}}(x, a)$, ignoring notification fatigue and opt-out guardrails.
5. **Ours — Closed-Loop Causal Uplift + Primal-Dual Optimizer ( + Distilled Student):** Maximizes causal conversion lift while dynamically adapting the shadow price $\lambda_t$ to strictly respect the opt-out guardrail $B_{\text{opt-out}}$.

### C. Target KPIs & Visual Outputs (Displayed Live in the Web UI)
* **The "Money Plot" — Multi-Objective Pareto Frontier:** Plots **Net Incremental Conversions (Y-axis)** against **User Opt-Out / Unsubscribe Rate % (X-axis)** across varying guardrail budgets. Demonstrates that our Primal-Dual Causal Orchestrator dominates the Pareto frontier—achieving **~90–95% of unconstrained peak conversion lift while reducing user opt-outs by >55% and cutting wasted message volume by >40%** compared to Greedy Predictive ML.
* **Closed-Loop Shadow-Price Trajectory Plot:** Shows how $\lambda_t$ automatically spikes during a simulated weekend traffic shift to keep the opt-out rate pinned safely at the $1.5\%$ ceiling, whereas a static-weight $\alpha$ baseline breaches the safety limit by $+80\%$.
* **Sub-Millisecond Distillation Benchmark:** Compares decision agreement ($>97\%$ fidelity) and latency ($<0.5\text{ ms}$ batch inference) between the full Teacher Solver and the distilled Student MLP.

---

## 5. Build & Deployment Specification for Cursor / Claude Code

Copy the instructions below directly into Cursor/Claude Code to generate the repository and deploy the live Cloud Run demo:

1. **Repository Structure (`decision-orchestrator-ops`):**
   - `app/main.py`: FastAPI backend serving REST endpoints (`/api/simulate`, `/api/pareto`, `/api/distill`, `/api/user-inspect`) and static frontend UI.
   - `app/data.py`: Automatic downloader/loader for the **Hillstrom 64k Multi-Channel dataset** (with deterministic fallback generator so tests run offline without network flakiness) + configurable Fatigue State-Space simulator.
   - `app/models.py`: CPU-fast **Causal Uplift estimators** (T-Learner / Doubly Robust CATE for conversion lift $\hat{\tau}_{\text{conv}}$ and fatigue risk $\hat{\tau}_{\text{fatigue}}$ using `scikit-learn` `HistGradientBoostingRegressor`), **Greedy Propensity baseline**, and **Teacher–Student PyTorch MLP distillation**.
   - `app/optimizer.py`: Vectorized NumPy **Lagrangian Primal-Dual Shadow-Price Solver** supporting global opt-out guardrails, per-channel capacity caps, and dynamic multi-day horizon simulation.
   - `app/ope.py`: **Doubly Robust (DR)** and **Inverse Propensity Scoring (IPS)** offline policy evaluation + Qini / Uplift curve calculators.
   - `static/index.html`: Executive-grade interactive dashboard (Tailwind CSS + Chart.js/Plotly) featuring:
     - Live Guardrail Sliders (*Max Allowed Opt-Out Rate %*, *Weekly Channel Budget*, *Traffic Shift Stress-Test*).
     - Real-time **Pareto Frontier Chart**, **Qini Uplift Curve**, and **Closed-Loop Shadow Price ($\lambda_t$) Tracker**.
     - Interactive **Single-User Inspector** showing how a *Persuadable*, *Sure Thing*, and *Sleeping Dog* user are routed differently, including an **In-App Audio Teaser** preview card.
   - `Dockerfile` & `tests/test_pipeline.py`: Ready for 1-command deployment to **Google Cloud Run** (`gcloud run deploy`).