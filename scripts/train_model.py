"""Train the LightGBM diagnosis model on DDXPlus.

Run from the curamate folder:  .venv/bin/python -m scripts.train_model
Produces artifacts/lgbm_model.txt and artifacts/answer_freq.json.

Input:  data/ddxplus/train.csv (15% used = ~154k patients) and validate.csv (5%, for early stopping)
Steps:  each patient -> 1 complete record + 2 simulated half-finished interviews (unknown answers = NaN)
        -> table X (461,922 rows x 905 numbers) + labels y (disease number 0-48)
        -> lgb.train (multiclass gradient boosting, early stopping) -> stopped at 110 rounds = 5,390 trees
Output: artifacts/lgbm_model.txt   the trained model (18.9 MB)
        artifacts/answer_freq.json P(yes | disease) table, used later to choose follow-up questions
"""
import json
import random
import time

import lightgbm as lgb
import numpy as np

import config
from core.features import Features


def build_matrix(f, patients, copies, rng, plaus):
    """Each patient -> 1 complete record + `copies` interview-style partial records (same label).

    Input:  list of parsed patients
    Output: X = rows of 905 numbers (the questions), y = disease numbers (the answers)
            e.g. 153,974 patients x 3 = 461,922 rows; the URTI patient gives 3 rows, all with y = 45
    """
    n_rows = len(patients) * (1 + copies)
    X = np.empty((n_rows, f.n_features), np.float32)
    y = np.empty(n_rows, np.int32)
    i = 0
    for age, sex, ans, init, pathology in patients:
        label = f.class_index[pathology]
        f.encode(age, sex, f.full_state(ans), out=X[i]); y[i] = label; i += 1                      # complete record
        for _ in range(copies):
            f.encode(age, sex, f.interview_state(ans, init, rng, plaus), out=X[i]); y[i] = label; i += 1   # partial copy
    return X, y


def answer_frequencies(f, patients):
    """P(question answered YES | disease) - used to estimate how likely each answer is.
    Output: 49 x 209 table, e.g. anaphylaxis -> rash 1.00, asthma -> rash 0.00. (Plain counting, not LightGBM.)"""
    counts = np.zeros((len(f.classes), len(f.roots)))
    totals = np.zeros(len(f.classes))
    ri = {r: i for i, r in enumerate(f.roots)}
    for _, _, ans, _, pathology in patients:
        k = f.class_index[pathology]
        totals[k] += 1
        for r in ans:
            counts[k, ri[r]] += 1
    return counts / np.maximum(totals[:, None], 1)


def main():
    config.ARTIFACTS.mkdir(exist_ok=True)
    f = Features(config.DATA_DIR)
    rng = random.Random(42)                       # fixed seed -> same patients and copies on every run
    plaus = f.plausible_roots()

    t = time.time()
    train = f.load_split(config.DATA_DIR / "train.csv", config.TRAIN_FRACTION, rng)
    val = f.load_split(config.DATA_DIR / "validate.csv", 0.05, rng)
    print(f"patients: train {len(train):,}  validation {len(val):,}  ({time.time() - t:.0f}s)", flush=True)

    t = time.time()
    X, y = build_matrix(f, train, config.INTERVIEW_COPIES, rng, plaus)
    Xv, yv = build_matrix(f, val, 1, rng, plaus)
    print(f"rows: train {X.shape[0]:,} x {X.shape[1]} features  ({time.time() - t:.0f}s, {X.nbytes / 1e9:.1f} GB)", flush=True)

    params = {
        "objective": "multiclass", "num_class": len(f.classes),                 # one score per disease + softmax
        "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 50,        # small trees -> no memorising
        "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,   # random subsets + penalty
        "num_threads": config.TRAIN_THREADS, "verbose": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X, y, free_raw_data=True)          # LightGBM bins each column (histograms) for speed
    dval = lgb.Dataset(Xv, yv, reference=dtrain)
    t = time.time()
    # The gradient boosting happens here: up to 200 rounds, each adding one tree per disease;
    # stops when validation log loss hasn't improved for 25 rounds.
    booster = lgb.train(params, dtrain, num_boost_round=200, valid_sets=[dval], valid_names=["validation"],
                        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(10)])
    print(f"trained {booster.best_iteration} rounds x {len(f.classes)} diseases in {time.time() - t:.0f}s", flush=True)
    booster.save_model(str(config.MODEL_FILE), num_iteration=booster.best_iteration)   # keep only the best rounds

    json.dump({"classes": f.classes, "roots": f.roots, "freq": answer_frequencies(f, train).round(4).tolist()},
              open(config.FREQ_FILE, "w"))
    print(f"saved {config.MODEL_FILE.name} ({config.MODEL_FILE.stat().st_size / 1e6:.1f} MB) and {config.FREQ_FILE.name}")


if __name__ == "__main__":
    main()
