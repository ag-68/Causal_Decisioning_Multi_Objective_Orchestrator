"""Hillstrom covariates and the declared four-channel fatigue response.

The public file supplies the shopping cards. The decision panel keeps those
cards and attaches a seeded response. Potential outcomes for Control, Email,
Push, and In-App Audio, the fatigue state, and opt-out labels come from the
formulas below. They are scenario settings. They are not fit from the cards.

Conversion probability
    sigmoid(baseline(x) + lift[type, a] * exp(-kappa * f))

Opt-out probability
    sigmoid(a0 + a1 * f + a2 * f^2 + cost[a] + log(shift))

Fatigue transition
    f <- gamma * f + beta[a] * 1{a != Control} + eps
"""

from __future__ import annotations

import gzip
import io
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = 0
EMAIL = 1
PUSH = 2
AUDIO = 3
ACTIONS = ("Control", "Email", "Push", "In-App Audio")
N_ACTIONS = 4

TYPE_SURE = 0
TYPE_PERSUADABLE = 1
TYPE_LOST = 2
TYPE_SLEEPING = 3
TYPE_NAMES = ("Sure Thing", "Persuadable", "Lost Cause", "Sleeping Dog")

SEGMENT_NAMES = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
SEGMENT_TO_CODE = {name: i for i, name in enumerate(SEGMENT_NAMES)}

ZIP_LEVELS = ("Rural", "Surburban", "Urban")
CHANNEL_LEVELS = ("Multichannel", "Phone", "Web")
F_INDEX = 5

# 0 fits the predictors. 1 chooses their settings. 2 is the untouched test.
SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2

# Rows: Sure Thing, Persuadable, Lost Cause, Sleeping Dog.
# Columns: Control, Email, Push, In-App Audio.
# Persuadable lifts stay ordered Audio < Email < Push.
# Sure-thing email and push stay near zero. Their audio lift is the light teaser.
# Sized so a rested Persuadable push is about 70% and a rested Sleeping Dog
# push is about 5%. Tiredness still shrinks the lift. Most other settings
# stay under 70%.
LIFT = np.array(
    [
        [0.0, 0.06, 0.04, 0.55],
        [0.0, 2.19, 3.54, 1.42],
        [0.0, 0.04, 0.02, 0.05],
        [0.0, -0.75, -1.67, -0.92],
    ],
    dtype=np.float64,
)
COST = np.array([0.0, 0.85, 1.75, 0.28], dtype=np.float64)
BETA = np.array([0.0, 0.40, 0.78, 0.20], dtype=np.float64)
GAMMA = 0.85
KAPPA = 0.65
OPT_A0 = -4.05
OPT_A1 = 0.90
OPT_A2 = 0.40
EPS_STD = 0.02
WEEKLY_TOUCHES = 3
F_TRAIN_HIGH = 3.0

HILLSTROM_URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
CACHE_NAME = "hillstrom.csv"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def cache_path() -> Path:
    return DATA_DIR / CACHE_NAME


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    required = [
        "recency",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
        "segment",
        "visit",
        "conversion",
        "spend",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Hillstrom table is missing columns: {missing}")
    for col in ("recency", "history", "mens", "womens", "newbie", "visit", "conversion", "spend"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["newbie"] = frame["newbie"].astype(int)
    frame["mens"] = frame["mens"].astype(int)
    frame["womens"] = frame["womens"].astype(int)
    frame["zip_code"] = frame["zip_code"].astype(str).replace({"Suburban": "Surburban"})
    frame["channel"] = frame["channel"].astype(str)
    frame["segment"] = frame["segment"].astype(str)
    known = frame["segment"].isin(SEGMENT_TO_CODE)
    return frame.loc[known].reset_index(drop=True)


def _download_hillstrom(dest: Path) -> pd.DataFrame:
    request = urllib.request.Request(HILLSTROM_URL, headers={"User-Agent": "orchestrator/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read()
    frame = pd.read_csv(io.BytesIO(gzip.decompress(payload)))
    frame = _coerce(frame)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(dest)
    return frame


def generate_fallback(n: int, seed: int) -> pd.DataFrame:
    """Seeded stand-in with the Hillstrom column layout and a weak email effect."""
    rng = np.random.default_rng(seed)
    history = np.clip(rng.lognormal(mean=5.0, sigma=0.65, size=n), 5.0, 4000.0)
    recency = rng.integers(1, 13, size=n)
    email = rng.integers(0, 3, size=n)
    logit = -4.3 + 0.12 * np.log1p(history) + 0.40 * (email > 0) - 0.03 * recency
    conversion = rng.random(n) < sigmoid(logit)
    visit_logit = -1.6 + 0.08 * np.log1p(history) + 0.30 * (email > 0)
    visit = rng.random(n) < sigmoid(visit_logit)
    spend = np.where(conversion, rng.uniform(8.0, 180.0, size=n), 0.0)
    zip_code = rng.choice(ZIP_LEVELS, size=n, p=[0.15, 0.45, 0.40])
    channel = rng.choice(CHANNEL_LEVELS, size=n, p=[0.12, 0.44, 0.44])
    return pd.DataFrame(
        {
            "recency": recency.astype(float),
            "history": history,
            "mens": rng.integers(0, 2, size=n),
            "womens": rng.integers(0, 2, size=n),
            "zip_code": zip_code,
            "newbie": rng.integers(0, 2, size=n),
            "channel": channel,
            "segment": [SEGMENT_NAMES[i] for i in email],
            "visit": visit.astype(int),
            "conversion": conversion.astype(int),
            "spend": spend,
        }
    )


def load_hillstrom(n_fallback: int, seed: int, fallback: bool) -> tuple[pd.DataFrame, str]:
    """Return the covariate log and a source label, ``hillstrom`` or ``fallback``."""
    path = cache_path()
    if not fallback and path.exists():
        return _coerce(pd.read_csv(path)), "hillstrom"
    if not fallback:
        try:
            return _download_hillstrom(path), "hillstrom"
        except (OSError, ValueError, urllib.error.URLError):
            pass
    return generate_fallback(n_fallback, seed), "fallback"


def assign_types(history: np.ndarray, recency: np.ndarray, newbie: np.ndarray) -> np.ndarray:
    """Map covariates to the four declared response types.

    Priority follows the scenario rules: newbie and low history is a Lost
    Cause; high history and recent purchase is a Sure Thing; stale mid-history
    users are Persuadable; remaining low-history recent users are Sleeping
    Dogs. Everyone else is filled with the same distinctions.
    """
    history = np.asarray(history, dtype=np.float64)
    recency = np.asarray(recency, dtype=np.float64)
    newbie = np.asarray(newbie, dtype=int)
    h_med = float(np.median(history))
    h_q25, h_q75 = np.quantile(history, [0.25, 0.75])
    r_med = float(np.median(recency))
    r_high = float(np.quantile(recency, 0.75))
    lost = (newbie == 1) & (history < h_med)
    sure = (history >= h_q75) & (recency <= r_med) & ~lost
    persu = (recency >= r_high) & (history >= h_q25) & (history <= h_q75) & ~lost & ~sure
    sleep = (history < h_med) & (recency <= r_med) & ~lost & ~sure & ~persu
    codes = np.full(len(history), -1, dtype=int)
    codes[lost] = TYPE_LOST
    codes[sure] = TYPE_SURE
    codes[persu] = TYPE_PERSUADABLE
    codes[sleep] = TYPE_SLEEPING
    rest = codes < 0
    codes[rest & (history >= h_med)] = TYPE_SURE
    rest = codes < 0
    codes[rest & (recency > r_med)] = TYPE_PERSUADABLE
    codes[codes < 0] = TYPE_SLEEPING
    return codes


def baseline_logit(df: pd.DataFrame) -> np.ndarray:
    history = np.log1p(df["history"].to_numpy(np.float64))
    recency = df["recency"].to_numpy(np.float64)
    mens = df["mens"].to_numpy(np.float64)
    womens = df["womens"].to_numpy(np.float64)
    return -3.40 + 0.70 * history - 0.28 * recency + 0.12 * mens + 0.10 * womens


def split_counts(n: int) -> tuple[int, int, int]:
    """Train, validation, and test sizes.

    On the full file, about 64,000 people, the cut is 48,000 to fit the
    predictors, 8,000 to choose their settings, and 8,000 held out. The held-out
    group is the only one the 21-day run, the Pareto chart, and the Qini curve
    may read. Once the settings are chosen, the 8,000 validation people join
    the fit. Smaller cohorts, including the test suite, keep the same shares:
    three quarters to train and one eighth each to validation and test.
    """
    if n >= 64_000:
        n_val = 8_000
        n_test = 8_000
    else:
        n_val = max(1, n // 8)
        n_test = max(1, n // 8)
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("cohort is too small to split")
    return n_train, n_val, n_test


def prepare_demo(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n > 0 and len(frame) > n:
        picked = rng.choice(len(frame), size=n, replace=False)
        demo = frame.iloc[picked].reset_index(drop=True)
    else:
        demo = frame.reset_index(drop=True)
    codes = assign_types(
        demo["history"].to_numpy(np.float64),
        demo["recency"].to_numpy(np.float64),
        demo["newbie"].to_numpy(int),
    )
    demo = demo.copy()
    demo["type_code"] = codes
    demo["type_name"] = [TYPE_NAMES[code] for code in codes]
    demo["baseline"] = baseline_logit(demo)
    return demo


def conversion_prob(
    baseline: np.ndarray,
    type_code: np.ndarray,
    actions: np.ndarray,
    fatigue: np.ndarray,
) -> np.ndarray:
    lift = LIFT[np.asarray(type_code, dtype=int), np.asarray(actions, dtype=int)]
    decay = np.exp(-KAPPA * np.maximum(np.asarray(fatigue, dtype=np.float64), 0.0))
    return sigmoid(np.asarray(baseline, dtype=np.float64) + lift * decay)


def optout_prob(actions: np.ndarray, fatigue: np.ndarray, shift: float = 1.0) -> np.ndarray:
    fatigue = np.maximum(np.asarray(fatigue, dtype=np.float64), 0.0)
    logit = (
        OPT_A0
        + OPT_A1 * fatigue
        + OPT_A2 * fatigue**2
        + COST[np.asarray(actions, dtype=int)]
        + np.log(max(float(shift), 1e-6))
    )
    return sigmoid(logit)


def fatigue_step(
    fatigue: np.ndarray,
    actions: np.ndarray,
    eps: np.ndarray | float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One fatigue transition. ``eps=0`` is the conditional mean."""
    fatigue = np.asarray(fatigue, dtype=np.float64)
    actions = np.asarray(actions, dtype=int)
    jump = BETA[actions] * (actions != CONTROL)
    if eps is None:
        if rng is None:
            noise = 0.0
        else:
            noise = rng.normal(0.0, EPS_STD, size=len(fatigue))
    else:
        noise = eps
    return GAMMA * fatigue + jump + noise


def make_cross_section(demo: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Randomized four-arm log with fatigue drawn in the training range."""
    rng = np.random.default_rng(seed + 17)
    n = len(demo)
    fatigue = rng.uniform(0.0, F_TRAIN_HIGH, size=n)
    actions = rng.integers(0, N_ACTIONS, size=n)
    p_conv = conversion_prob(demo["baseline"].to_numpy(), demo["type_code"].to_numpy(), actions, fatigue)
    p_opt = optout_prob(actions, fatigue, shift=1.0)
    out = demo.copy()
    out["f"] = fatigue
    out["action"] = actions
    out["y_conversion"] = (rng.random(n) < p_conv).astype(int)
    out["y_optout"] = (rng.random(n) < p_opt).astype(int)
    perm = rng.permutation(n)
    n_train, n_val, _n_test = split_counts(n)
    split = np.empty(n, dtype=int)
    split[perm[:n_train]] = SPLIT_TRAIN
    split[perm[n_train : n_train + n_val]] = SPLIT_VAL
    split[perm[n_train + n_val :]] = SPLIT_TEST
    out["split"] = split
    return out


def feature_matrix(df: pd.DataFrame, fatigue: np.ndarray) -> np.ndarray:
    """Covariates plus fatigue. Column ``F_INDEX`` is the fatigue state."""
    n = len(df)
    fatigue = np.asarray(fatigue, dtype=np.float64)
    if len(fatigue) != n:
        raise ValueError("fatigue length does not match the frame")
    base = np.column_stack(
        [
            df["recency"].to_numpy(np.float64),
            np.log1p(df["history"].to_numpy(np.float64)),
            df["mens"].to_numpy(np.float64),
            df["womens"].to_numpy(np.float64),
            df["newbie"].to_numpy(np.float64),
            fatigue,
        ]
    )
    zip_code = df["zip_code"].to_numpy()
    channel = df["channel"].to_numpy()
    zip_oh = np.column_stack([(zip_code == level).astype(np.float64) for level in ZIP_LEVELS])
    channel_oh = np.column_stack([(channel == level).astype(np.float64) for level in CHANNEL_LEVELS])
    return np.hstack([base, zip_oh, channel_oh])


def covariate_matrix(df: pd.DataFrame) -> np.ndarray:
    """Hillstrom features with the fatigue column removed."""
    packed = feature_matrix(df, np.zeros(len(df), dtype=np.float64))
    return np.delete(packed, F_INDEX, axis=1)


def bitrate_illustration() -> dict:
    """Serving-hook sketch. The index is not the output of a trained codec."""
    kbps = [6, 12, 24]
    expressiveness = [0.55, 0.78, 0.93]
    return {
        "kbps": kbps,
        "expressiveness": expressiveness,
        "caption": (
            "Illustrative codebook rate versus an expressiveness index "
            "at 6 kbps and 24 kbps. Not a trained neural codec or TTS model."
        ),
    }
