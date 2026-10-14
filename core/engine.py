"""LightGBM diagnosis + adaptive questioning (information gain) + stopping rule.

Loads the trained model (artifacts/lgbm_model.txt) and the yes-frequency table (artifacts/answer_freq.json),
then provides everything the interview needs:
  predict()         905-number rows -> 49 disease percentages
  rank_questions()  which follow-up question to ask next (biggest expected drop in uncertainty)
  stop_check()      whether the interview can stop
  key_findings()    which answers pushed the diagnosis up or down (for the report)
LightGBM runs inside this Python process - no network calls here.
"""
import json

import lightgbm as lgb
import numpy as np

import config


def entropy(P):
    """Uncertainty in bits: 0 = one disease at 100%, +1 bit = twice as many realistic suspects.
    Example: 71% / 22% / 7% -> 1.10 bits (about 2.1 diseases still in the running)."""
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log2(P)).sum(axis=-1)


class Engine:
    def __init__(self, features):
        """Input: a Features object. Loads the 5,390 trees and the P(yes | disease) table."""
        self.f = features
        self.booster = lgb.Booster(model_file=str(config.MODEL_FILE))   # load the trained trees (no training here)
        meta = json.load(open(config.FREQ_FILE))
        if meta["classes"] != features.classes:
            raise ValueError("answer_freq.json does not match the dataset's disease list - retrain")
        self.freq = np.array(meta["freq"])                       # [disease, question] = P(yes | disease)
        self.freq_col = {r: i for i, r in enumerate(meta["roots"])}
        self.n_rounds = self.booster.num_trees() // len(features.classes)   # 5,390 / 49 = 110 rounds

    # ---------- prediction ----------
    def predict(self, X):
        """Input: one or more 905-number rows. Output: 49 percentages per row (sum = 1).
        Runs every row through all 5,390 trees, adds up each disease's leaves, applies softmax (~0.5 ms)."""
        return self.booster.predict(np.atleast_2d(X))

    def differential(self, p, n=8):
        """49 percentages -> top n as [{"disease": "Pulmonary embolism", "prob": 0.992, "severity": 2}, ...]."""
        f = self.f
        return [{"disease": f.classes[i], "prob": float(p[i]), "severity": int(f.severity[i])}
                for i in np.argsort(-p)[:n]]

    # ---------- choosing the next question ----------
    def candidates(self, state, excluded, p):
        """Unasked yes/no questions that appear on the top suspects' disease lists.
        Input: current answers, "don't know" questions, current percentages. Output: ~40 question codes."""
        f = self.f
        pool = [r for r in f.binary_roots if r not in state and r not in excluded]
        top = np.argsort(-p)[:config.TOP_SUSPECTS]
        relevant = set().union(*(f.disease_roots[f.classes[i]] for i in top))
        return [r for r in pool if r in relevant] or pool

    def rank_questions(self, age, sex, state, excluded, p):
        """What-if: predict with YES and with NO for every candidate; score = expected drop in uncertainty.

        Input:  patient's age/sex, current answers (state), excluded questions, current percentages p
        Output: candidate questions sorted best-first, e.g.
                [{"code": "E_196", "text": "Have you had surgery within the last month?", "gain": 0.65,
                  "p_yes": 0.21, "if_yes": {"disease": "Pulmonary embolism", "prob": 0.77}, "if_no": {...}}, ...]
        Formula: gain = H(now) - [P(yes) x H(if yes) + P(no) x H(if no)]
        """
        f = self.f
        cands = self.candidates(state, excluded, p)
        if not cands:
            return []
        X = np.empty((2 * len(cands), f.n_features), np.float32)
        for i, r in enumerate(cands):                                          # 2 pretend copies per question
            f.encode(age, sex, {**state, r: ([r], set())}, out=X[2 * i])      # YES, follow-up details unknown
            f.encode(age, sex, {**state, r: False}, out=X[2 * i + 1])          # NO
        P = self.predict(X)                                                    # all copies in one batch (~15 ms)
        p_yes = np.array([p @ self.freq[:, self.freq_col[r]] for r in cands])  # sum_d P(d) * P(yes | d)
        h_now = entropy(p)
        h_yes, h_no = entropy(P[0::2]), entropy(P[1::2])
        gain = h_now - (p_yes * h_yes + (1 - p_yes) * h_no)                    # information gain
        ranked = []
        for i in np.argsort(-gain):
            py, pn = P[2 * i], P[2 * i + 1]
            ranked.append({
                "code": cands[i], "text": f.question(cands[i]), "gain": float(gain[i]), "p_yes": float(p_yes[i]),
                "if_yes": {"disease": f.classes[int(py.argmax())], "prob": float(py.max())},
                "if_no": {"disease": f.classes[int(pn.argmax())], "prob": float(pn.max())},
            })
        return ranked

    # ---------- stopping rule ----------
    def stop_check(self, p, n_asked):
        """Input: 49 percentages + number of questions asked. Output: (stop?, reason shown on the website).
        Rules in order: 15 questions -> stop; top < 90% -> continue; < 3 questions -> continue;
        another emergency (severity 1-2) above 1% -> continue; otherwise stop. Thresholds live in config.py."""
        f = self.f
        k = int(np.argmax(p))
        open_emergencies = [f.classes[i] for i in np.argsort(-p)
                            if i != k and f.severity[i] <= 2 and p[i] > config.EMERGENCY_THRESHOLD]
        if n_asked >= config.MAX_QUESTIONS:
            return True, f"Reached the maximum of {config.MAX_QUESTIONS} follow-up questions."
        if p[k] < config.STOP_CONFIDENCE:
            return False, f"Top diagnosis is {p[k]:.0%} - need {config.STOP_CONFIDENCE:.0%}."
        if n_asked < config.MIN_QUESTIONS:
            return False, f"Asking at least {config.MIN_QUESTIONS} follow-up questions before concluding."
        if open_emergencies:
            return False, f"Still ruling out emergency: {open_emergencies[0]} ({p[f.class_index[open_emergencies[0]]]:.1%})."
        return True, f"{f.classes[k]} at {p[k]:.0%} and no other emergency above {config.EMERGENCY_THRESHOLD:.0%}."

    # ---------- explanation ----------
    def key_findings(self, x, state, k, top=6):
        """Which known answers pushed disease k's score up or down (LightGBM's per-feature contributions).
        Output example: [{"code": "E_196", "answer": "yes", "effect": 2.1}, ...]  (+ = pushed it up)"""
        f = self.f
        F = f.n_features
        # pred_contrib=True: LightGBM splits each disease's score into "how much each column added"
        contrib = self.booster.predict(x[None], pred_contrib=True)[0][k * (F + 1):(k + 1) * (F + 1) - 1]
        per_root = {}
        for j, value in enumerate(contrib):
            root = f.col_root[j]
            if root in state or root in ("AGE", "SEX_F"):          # only answers the doctor actually gave
                per_root[root] = per_root.get(root, 0.0) + float(value)
        items = sorted(per_root.items(), key=lambda kv: -abs(kv[1]))[:top]
        return [{"code": r, "answer": ("no" if state.get(r) is False else "yes") if r in state else "",
                 "effect": round(v, 2)} for r, v in items]
