"""Evaluate the trained model on unseen test.csv patients.

  1. Diagnosis accuracy when the doctor enters all / half / a third / only the chief complaint.
  2. Full interview simulation: start from the chief complaint, let the engine pick questions, answer them
     truthfully from the patient's record, stop with the real stopping rule.

Run from the curamate folder:  .venv/bin/python -m scripts.evaluate
Input:  data/ddxplus/test.csv (5% sample, never seen in training) + the trained model
Output: printed results + artifacts/eval.json
        (last run: 99.7% top-1 with all findings; full interview 97.0% top-1 with 3.8 questions on average)
"""
import json
import random
import time

import numpy as np

import config
from core.engine import Engine
from core.features import Features


def volunteered(f, ans, init, keep, rng):
    """Doctor entered only some of the patient's findings; everything else is UNKNOWN."""
    init_root = f.root_of[init]
    return {r: (t, None) for r, t in ans.items() if r == init_root or rng.random() < keep}


def accuracy(f, eng, patients, keep, rng):
    """Input: test patients + share of findings entered (keep). Output: {"top1": ..., "top5": ..., "emergency_top5": ...}."""
    X = np.vstack([f.encode(a, s, volunteered(f, ans, i, keep, rng)) for a, s, ans, i, _ in patients])
    y = np.array([f.class_index[p] for *_, p in patients])
    top5 = np.argsort(-eng.predict(X), axis=1)[:, :5]
    emergency = f.severity[y] <= 2
    hit5 = (top5 == y[:, None]).any(1)
    return {"top1": float((top5[:, 0] == y).mean()), "top5": float(hit5.mean()), "emergency_top5": float(hit5[emergency].mean())}


def interview(f, eng, patient):
    """Simulates one full interview, exactly like the app: predict -> stop? -> best question -> true answer -> repeat.
    Output: (questions asked, correct #1?, correct in top 5?, is it an emergency?)."""
    age, sex, ans, init, pathology = patient
    root = f.root_of[init]
    state, n = {root: (ans[root], None)}, 0                 # start with only the chief complaint
    while True:
        p = eng.predict(f.encode(age, sex, state))[0]
        stop, _ = eng.stop_check(p, n)
        if stop:
            break
        ranked = eng.rank_questions(age, sex, state, set(), p)
        if not ranked:
            break
        q = ranked[0]["code"]
        state[q] = (ans[q], None) if q in ans else False    # answer truthfully from the patient's record
        n += 1
    order = np.argsort(-p)
    y = f.class_index[pathology]
    return n, order[0] == y, y in order[:5], f.severity[y] <= 2


def main():
    f = Features(config.DATA_DIR)
    eng = Engine(f)
    rng = random.Random(7)
    test = f.load_split(config.DATA_DIR / "test.csv", 0.05, rng)
    results = {"test_patients": len(test), "model_rounds": eng.n_rounds}

    print(f"Diagnosis accuracy on {len(test):,} unseen test patients")
    print(f"{'doctor enters':<26}{'top-1':>8}{'top-5':>8}{'emergency top-5':>18}")
    results["accuracy"] = {}
    for name, keep in [("all findings", 1.0), ("~50% of findings", 0.5), ("~30% of findings", 0.3), ("chief complaint only", 0.0)]:
        r = accuracy(f, eng, test, keep, rng)
        results["accuracy"][name] = r
        print(f"{name:<26}{r['top1']:>8.1%}{r['top5']:>8.1%}{r['emergency_top5']:>18.1%}")

    sample = rng.sample(test, 300)
    t = time.time()
    n, t1, t5, em = map(np.array, zip(*[interview(f, eng, p) for p in sample]))
    results["interview"] = {"patients": len(sample), "avg_questions": float(n.mean()), "median_questions": float(np.median(n)),
                            "top1": float(t1.mean()), "top5": float(t5.mean()), "emergency_top1": float(t1[em].mean()),
                            "emergency_top5": float(t5[em].mean()), "seconds": round(time.time() - t, 1)}
    i = results["interview"]
    print(f"\nInterview simulation on {len(sample)} patients (start: chief complaint only)")
    print(f"avg questions {i['avg_questions']:.1f} (median {i['median_questions']:.0f}) | top-1 {i['top1']:.1%} | "
          f"top-5 {i['top5']:.1%} | emergencies top-1 {i['emergency_top1']:.1%}, top-5 {i['emergency_top5']:.1%}")
    json.dump(results, open(config.ARTIFACTS / "eval.json", "w"), indent=2)
    print(f"saved {config.ARTIFACTS / 'eval.json'}")


if __name__ == "__main__":
    main()
