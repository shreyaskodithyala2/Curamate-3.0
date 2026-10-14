"""Turns patient answers into the 905 numbers LightGBM reads.

Every question is YES, NO or UNKNOWN. UNKNOWN is stored as NaN, and LightGBM learns at each
split which way unknown values should go. Used by both training and the web app.

What this file does:
  1. Reads the DDXPlus dictionaries (release_evidences.json = 223 questions, release_conditions.json = 49 diseases)
     and designs the 905 input columns: 208 yes/no questions + 6 scales (0-10) + 689 answer options + age + sex.
  2. Reads and cleans train/validate/test CSV rows.
  3. encode(): turns a patient's answers into one row of 905 numbers (1 = yes, 0 = no, NaN = unknown).
  4. Creates realistic half-finished interviews from complete records (training data augmentation).

Used by: scripts/train_model.py, scripts/evaluate.py, core/engine.py and app.py - so a patient is always
encoded exactly the same way in training and in the live app.
"""
import csv
import json

import numpy as np

csv.field_size_limit(10**9)

# Order of follow-up fields shown after a YES to pain / skin lesions / swelling.
CHILD_ORDER = {
    "E_53": ["E_55", "E_54", "E_56", "E_59", "E_58", "E_57"],
    "E_129": ["E_133", "E_130", "E_136", "E_134", "E_132", "E_135", "E_131"],
    "E_151": ["E_152"],
}
# Labels for the two ends of each 0-10 slider on the website's detail form.
SCALE_HINTS = {
    "E_56": ("no pain", "worst pain"),
    "E_58": ("diffuse", "one precise spot"),
    "E_59": ("very gradual", "sudden"),
    "E_132": ("not swollen", "very swollen"),
    "E_134": ("no pain", "very painful"),
    "E_136": ("no itch", "severe itch"),
}
YES_NO_LABELS = {"Y": "Yes", "N": "No", "NA": "None"}


class Features:
    def __init__(self, data_dir):
        """Reads the two dataset dictionaries and builds every lookup table the project uses.

        Input:  folder containing release_evidences.json and release_conditions.json
        Output: nothing returned; sets e.g.
                self.cols        -> the 905 column names (["E_91", "E_55_@_V_14", ..., "AGE", "SEX_F"])
                self.classes     -> the 49 disease names (alphabetical; index = LightGBM class number)
                self.root_of     -> question -> its main question  (root_of["E_55"] = "E_53": pain location belongs to "pain?")
                self.disease_roots -> disease -> the questions on its list (used to pick follow-up questions)
        """
        self.ev = json.load(open(data_dir / "release_evidences.json", encoding="utf-8"))
        self.cond = json.load(open(data_dir / "release_conditions.json", encoding="utf-8"))
        ev = self.ev

        # 5 evidences point to parent codes that are missing from the file -> treat them as standalone
        self.root_of = {k: (v["code_question"] if v["code_question"] in ev else k) for k, v in ev.items()}
        self.roots = [k for k in ev if self.root_of[k] == k]                     # 209 main questions
        self.children = {r: [k for k in ev if self.root_of[k] == r and k != r] for r in self.roots}   # follow-ups
        self.scale = {k for k, v in ev.items() if v["data_type"] == "C" and isinstance(v["possible-values"][0], int)}  # the six 0-10 questions

        # Columns: one per yes/no question or 0-10 scale, one per non-default option of the others
        # e.g. "E_91" (fever), "E_56" (pain intensity 0-10), "E_55_@_V_89" (pain location = forehead)
        cols = []
        for k, v in ev.items():
            if v["data_type"] == "B" or k in self.scale:
                cols.append(k)
            else:
                cols += [f"{k}_@_{x}" for x in v["possible-values"] if str(x) != str(v["default_value"])]
        self.cols = cols + ["AGE", "SEX_F"]                                      # 905 columns in total
        self.n_features = len(self.cols)
        self.ci = {c: i for i, c in enumerate(self.cols)}                        # column name -> position
        self.q_cols = {k: [] for k in ev}                                        # question -> all its column positions
        for c in cols:
            self.q_cols[c.split("_@_")[0]].append(self.ci[c])
        self.col_root = [c if c in ("AGE", "SEX_F") else self.root_of[c.split("_@_")[0]] for c in self.cols]

        self.classes = sorted(self.cond)                                         # 49 diseases
        self.class_index = {c: i for i, c in enumerate(self.classes)}            # "URTI" -> 45
        self.severity = np.array([self.cond[c]["severity"] for c in self.classes])   # 1 = emergency ... 5 = mild
        self.disease_roots = {c: {self.root_of[k] for k in list(v["symptoms"]) + list(v["antecedents"])}
                              for c, v in self.cond.items()}
        self.binary_roots = [r for r in self.roots if ev[r]["data_type"] == "B"]  # 208 askable yes/no questions

    # ---------- labels ----------
    def question(self, code):
        """"E_91" -> "Do you have a fever (either felt or measured with a thermometer)?" """
        return self.ev[code]["question_en"]

    def value_label(self, code, value):
        """Answer code -> readable word, e.g. ("E_55", "V_89") -> "forehead"."""
        meaning = self.ev[code]["value_meaning"].get(str(value), {}).get("en", str(value))
        return YES_NO_LABELS.get(meaning, meaning)

    # ---------- dataset rows ----------
    def is_default(self, tok):
        """True for "nothing to report" answers like E_57_@_V_123 (pain spreads: nowhere) - not real symptoms."""
        b, _, v = tok.partition("_@_")
        return v != "" and str(self.ev[b]["default_value"]) == v

    def parse_row(self, row):
        """CSV row -> (age, sex, {root: [tokens]}, initial evidence, pathology). Default 'none' values dropped.

        Example output:
          (18, "M", {"E_91": ["E_91"], "E_53": ["E_53", "E_55_@_V_89", "E_56_@_4", ...], ...}, "E_91", "URTI")
          -> fever yes, pain yes (+ its details grouped under E_53), chief complaint E_91, real disease URTI
        """
        toks = [t.strip(" '") for t in row["EVIDENCES"].strip("[]").split(",")]   # text "['E_48', ...]" -> list
        ans = {}
        for t in toks:
            if not self.is_default(t):
                ans.setdefault(self.root_of[t.split("_@_")[0]], []).append(t)       # group details under main question
        return int(row["AGE"]), row["SEX"], ans, row["INITIAL_EVIDENCE"], row["PATHOLOGY"]

    def load_split(self, path, fraction, rng):
        """Reads a CSV and keeps a random `fraction` of its patients (e.g. 0.15 of train.csv -> ~154k patients).
        Output: list of parse_row() tuples."""
        with open(path, encoding="utf-8") as f:
            return [self.parse_row(r) for r in csv.DictReader(f) if rng.random() < fraction]

    # ---------- encoding ----------
    def encode(self, age, sex, state, out=None):
        """state: {root: False | (tokens, known_children)}. False = NO.
        (tokens, known) = YES; known = set of follow-up questions whose answers are known (None = all).
        Roots not in state are UNKNOWN (NaN).

        Input:  age, sex, state - e.g. {"E_91": (["E_91"], None), "E_66": False}  (fever yes, short of breath no)
        Output: numpy array of 905 numbers - the exact row LightGBM reads, e.g.
                E_91 = 1 (yes), E_66 = 0 (no), every unanswered column = NaN (unknown), AGE = 34, SEX_F = 1
        """
        x = out if out is not None else np.empty(self.n_features, np.float32)
        x.fill(np.nan)                              # start with everything UNKNOWN
        x[-2] = age
        x[-1] = 1.0 if sex == "F" else 0.0
        for root, val in state.items():
            if val is False:                        # NO -> the question and all its follow-up columns = 0
                for q in [root] + self.children[root]:
                    x[self.q_cols[q]] = 0.0
                continue
            tokens, known = val                     # YES
            for q in [root] + self.children[root]:
                if q == root or known is None or q in known:
                    x[self.q_cols[q]] = 0.0         # known follow-ups start at 0; unasked ones stay NaN
            for t in tokens:
                b, _, v = t.partition("_@_")
                if b in self.scale:
                    x[self.ci[b]] = float(v)        # scale answer, e.g. "E_56_@_7" -> column E_56 = 7
                else:
                    x[self.ci[t]] = 1.0             # yes / chosen option -> 1
        return x

    # ---------- training-time simulation ----------
    def full_state(self, ans):
        """Complete record: every question known (the patient's are YES, all others NO).
        Example: the URTI patient -> 9 YES, 200 NO, 0 unknown."""
        state = {r: False for r in self.roots}
        state.update({r: (toks, None) for r, toks in ans.items()})
        return state

    def plausible_roots(self):
        """For each question: every question that shares a disease list with it.
        (= what a doctor might plausibly ask next; used to simulate realistic interviews)"""
        lists = list(self.disease_roots.values())
        return {r: sorted(set().union(*[l for l in lists if r in l]) or {r}) for r in self.roots}

    def interview_state(self, ans, init, rng, plaus):
        """A realistic half-finished interview: the patient volunteered some true symptoms, the doctor asked
        some questions chosen WITHOUT knowing the disease and recorded the TRUE answer. Rest UNKNOWN.

        Why: the dataset only has complete records, but real doctors enter partial information.
        Training on these copies teaches the model to diagnose from incomplete input.
        Example output for the URTI patient: YES fever, sore throat, cough, smoker, sweating; 18 NO; 186 UNKNOWN.
        """
        init_root = self.root_of[init]
        keep = rng.uniform(0.0, 0.8)                                       # patient mentions 0-80% of symptoms
        state = {r: self._hide_some_details(r, toks, rng) for r, toks in ans.items()
                 if r == init_root or rng.random() < keep}                 # chief complaint is always kept
        asked = rng.sample(self.roots, rng.randint(0, 25))                 # doctor asks 0-25 random questions...
        asked += rng.sample(plaus[init_root], min(rng.randint(0, 15), len(plaus[init_root])))   # ...+ 0-15 related ones
        for r in asked:
            state[r] = self._hide_some_details(r, ans[r], rng) if r in ans else False   # record the TRUE answer
        return state

    def _hide_some_details(self, root, toks, rng):
        """Sometimes the doctor skips the pain/rash/swelling detail form, or parts of it."""
        kids = self.children[root]
        if not kids:
            return toks, None
        known = set() if rng.random() < 0.3 else {k for k in kids if rng.random() > 0.15}   # skip form 30%, each detail 15%
        toks = [t for t in toks if t.split("_@_")[0] == root or t.split("_@_")[0] in known]
        return toks, known
