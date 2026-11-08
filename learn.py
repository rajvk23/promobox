"""NOVELTY 1 - self-supervised hook / cliffhanger models.

Idea: in a professionally written show every episode ENDS on a writer-chosen cliffhanger and OPENS on a writer-chosen
hook. So the provided episodes contain free labels:
    cliffhanger model : positive = last ~6 s of an episode            negative = last ~6 s before any other event end
    hook model        : positive = first 3 s of an episode            negative = first 3 s from any other event start
No hand labelling. The model is show-agnostic by construction: character names are masked and explicit
"cliffhanger"/"cut to black" stage directions (which would leak the label) are removed before featurising.

Honest limits: it learns what THIS show's writers do at episode edges; it is evidence of "cliffhanger style",
not of what performs best on Instagram. Mid-episode moments that are cliffhangers but not at an edge are labelled negative (label noise).

    python learn.py --input "<folder of episodes>"        # trains, prints leave-episode-out AUC, saves models/*.pkl
"""
import argparse
import json
import os
import pickle
import random
import re

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import promo_finder as pf

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LEAK = re.compile(r"cut to black|cliffhanger|fade (out|to)|smash cut|end of episode|to be continued", re.I)
GENERIC = {"ACTION", "CAMERA", "SOUND", "FX", "THE", "A", "AN", "OF", "MAN", "WOMAN", "LORD", "LADY", "GUARD", "ROYAL"}


# ------------------------------------------------------------------ edge extraction (same code for train & inference)
def story(evs):
    return [e for e in evs if e.kind in ("DIALOGUE", "ACTION") and not LEAK.search(e.text)]


def hook_events(ep_events, start):
    return [e for e in story(ep_events) if e.start < start + 3.0 and e.end > start - 1e-6 and e.start >= start - 1e-6]


def cliff_events(ep_events, end):
    return [e for e in story(ep_events) if e.end > end - 6.0 and e.start < end]


def names_of(events):
    names = set()
    for e in events:
        if e.kind == "DIALOGUE":
            for t in re.findall(r"[A-Za-z']+", e.speaker):
                if t.upper() not in GENERIC and len(t) > 2:
                    names.add(t.lower())
    return names


def mask(text, names):
    toks = re.findall(r"[A-Za-z']+|[.?!…—-]+|\.\.\.", text)
    return " ".join("NAME" if t.lower() in names else t.lower() for t in toks)


def edge_text(evs, names):
    parts = []
    for e in evs:
        emo = " ".join("emo_" + w.lower() for w in re.findall(r"[A-Za-z]{3,}", e.emotion)) if e.kind == "DIALOGUE" else ""
        parts.append(f"{emo} {mask(e.text, names)}")
    return " ".join(parts)


YOU = re.compile(r"\byou(r|rs)?\b", re.I)


def hand_features(evs):
    """Show-agnostic structural features."""
    if not evs:
        return [0.0] * 14
    txt = " ".join(e.text for e in evs)
    dl = [e for e in evs if e.kind == "DIALOGUE"]
    last = evs[-1]
    w = pf.words(txt)
    # NOTE: no event-kind / event-count features on purpose (they encode "edge of episode" style, not hook strength)
    return [
        len(dl), len(w),
        txt.count("!"), txt.count("?"), txt.count("...") + txt.count("—"),
        sum(pf.emo_score(e.emotion) for e in dl),
        pf.kw_hits(txt, pf.HOOK_WORDS), pf.kw_hits(txt, pf.REVEAL_PHRASES),
        len(YOU.findall(txt)),
        float(last.text.rstrip().endswith("?")),
        float(last.text.rstrip().endswith(("...", "—", "-"))), len(pf.words(last.text)),
        float(bool(dl) and pf.words(dl[0].text)[:1] != [] and pf.words(dl[0].text)[0] in pf.WEAK_OPENERS),
        float(any(pf.kw_hits(e.text.lower(), pf.RESOLUTION_WORDS) for e in evs)),
    ]


# ------------------------------------------------------------------ dataset
def build_dataset(episodes, seed=0, neg_per_ep=40):
    rnd = random.Random(seed)
    data = {"hook": [], "cliff": []}   # (text, hand, label, group)
    pos_h, pos_c = [], []
    for ep in episodes:
        nm = names_of(ep.events)
        s = story(ep.events)
        if len(s) < 6:
            continue
        # positives
        hev = hook_events(ep.events, s[0].start)
        last_end = max(e.end for e in s)
        cev = cliff_events(ep.events, last_end)
        # Many endings in this show are tender/closing, not cliffhangers (found by reading them), so "episode edge" alone is a
        # noisy label. Keep only the edges that the generic rules also rate above the median (confident positives);
        # the learned model then generalises BEYOND the rules' keyword lists.
        hs = pf.score_hook([e for e in ep.events if e.end > s[0].start], s[0].start)[0]
        cs = pf.score_cliff([e for e in ep.events if e.end > last_end - 6], last_end, ep)[0]
        pos_h.append((hs, (edge_text(hev, nm), hand_features(hev), 1, ep.number)))
        pos_c.append((cs, (edge_text(cev, nm), hand_features(cev), 1, ep.number)))
        # negatives, MATCHED on event kind (dialogue/action) of the edge event. Without this the model just learns
        # "episodes open/close on a scene-setting ACTION line", a confound we found in the first version.
        k_first = hev[0].kind if hev else "ACTION"
        k_last = cev[-1].kind if cev else "ACTION"
        starts = [e.start for e in s if e.kind == k_first and 8 <= e.start <= ep.duration - 25]
        ends = [e.end for e in s if e.kind == k_last and 10 <= e.end <= ep.duration - 8]
        for st in rnd.sample(starts, min(neg_per_ep, len(starts))):
            ev = hook_events(ep.events, st)
            if ev:
                data["hook"].append((edge_text(ev, nm), hand_features(ev), 0, ep.number))
        for en in rnd.sample(ends, min(neg_per_ep, len(ends))):
            ev = cliff_events(ep.events, en)
            if ev:
                data["cliff"].append((edge_text(ev, nm), hand_features(ev), 0, ep.number))
    for pos, key in ((pos_h, "hook"), (pos_c, "cliff")):
        med = float(np.median([p[0] for p in pos]))
        data[key] += [p[1] for p in pos if p[0] >= med]
    return data


class EdgeModel:
    def __init__(self, variant="both"):
        self.variant = variant
        self.tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=4000)
        self.sc = StandardScaler()
        self.lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)

    def _x(self, texts, hand, fit=False):
        H = np.log1p(np.abs(np.array(hand, dtype=float)))
        if fit:
            H = self.sc.fit_transform(H); T = self.tf.fit_transform(texts)
        else:
            H = self.sc.transform(H); T = self.tf.transform(texts)
        if self.variant == "hand":
            return csr_matrix(H)
        if self.variant == "tfidf":
            return T
        return hstack([T, csr_matrix(H)]).tocsr()

    def fit(self, texts, hand, y):
        self.lr.fit(self._x(texts, hand, fit=True), y)
        return self

    def proba(self, texts, hand):
        return self.lr.predict_proba(self._x(texts, hand))[:, 1]


def cv_auc(rows, variant, folds=5):
    texts = [r[0] for r in rows]; hand = [r[1] for r in rows]; y = np.array([r[2] for r in rows]); g = np.array([r[3] for r in rows])
    aucs = []
    for tr, te in GroupKFold(folds).split(texts, y, g):
        m = EdgeModel(variant).fit([texts[i] for i in tr], [hand[i] for i in tr], y[tr])
        p = m.proba([texts[i] for i in te], [hand[i] for i in te])
        if len(set(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def train(input_dir, verbose=True):
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".txt"))
    eps = [pf.parse_episode(os.path.join(input_dir, f)) for f in files]
    data = build_dataset(eps)
    os.makedirs(MODEL_DIR, exist_ok=True)
    report = {"episodes": len(eps)}
    for kind in ("hook", "cliff"):
        rows = data[kind]
        pos = sum(r[2] for r in rows)
        report[kind] = {"n": len(rows), "positives": pos}
        for variant in ("hand", "tfidf", "both"):
            m, s = cv_auc(rows, variant)
            report[kind][variant] = {"leave_episode_out_auc": round(m, 3), "std": round(s, 3)}
        # SHIPPED MODEL = show-agnostic 'hand' features only. TF-IDF scores a bit higher in cross-validation, but its best
        # tokens are plot-specific ("ten days", "choose your"): plot threads span episodes, so leave-episode-out CV flatters it
        # and it would not transfer to a new show. 'both' is trained only to report which tokens it latched onto.
        final = EdgeModel("hand").fit([r[0] for r in rows], [r[1] for r in rows], np.array([r[2] for r in rows]))
        pickle.dump(final, open(os.path.join(MODEL_DIR, f"{kind}.pkl"), "wb"))
        diag = EdgeModel("both").fit([r[0] for r in rows], [r[1] for r in rows], np.array([r[2] for r in rows]))
        names = list(diag.tf.get_feature_names_out())
        coef = diag.lr.coef_[0][:len(names)]
        order = np.argsort(coef)
        report[kind]["shipped_variant"] = "hand"
        report[kind]["tfidf_top_positive_tokens(diagnostic)"] = [names[i] for i in order[::-1][:15]]
        hn = ["n_dialogue", "n_words", "excl", "question", "ellipsis/dash", "emotion_intensity", "hook_words", "reveal_phrases",
              "you_count", "last_is_question", "last_trails_off", "last_len", "weak_opener", "resolution_words"]
        hc = final.lr.coef_[0]
        report[kind]["hand_feature_weights"] = {n: round(float(w), 2) for n, w in zip(hn, hc)}
    json.dump(report, open(os.path.join(MODEL_DIR, "cv_report.json"), "w"), indent=2)
    if verbose:
        print(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------ inference
_cache = {}


def load():
    if "m" not in _cache:
        try:
            _cache["m"] = {k: pickle.load(open(os.path.join(MODEL_DIR, f"{k}.pkl"), "rb")) for k in ("hook", "cliff")}
        except Exception:
            _cache["m"] = None
    return _cache["m"]


def score_windows(ep, cands):
    """Fill c.clf = {'hook': 0-10, 'cliff': 0-10} for every candidate. Returns False if no trained model."""
    models = load()
    if not models:
        return False
    nm = names_of(ep.events)
    ht, hh, ct, ch = [], [], [], []
    for c in cands:
        he = hook_events(ep.events, c.start); ce = cliff_events(ep.events, c.end)
        ht.append(edge_text(he, nm)); hh.append(hand_features(he))
        ct.append(edge_text(ce, nm)); ch.append(hand_features(ce))
    ph = models["hook"].proba(ht, hh); pc = models["cliff"].proba(ct, ch)
    for c, a, b in zip(cands, ph, pc):
        c.clf = {"hook": round(10 * float(a), 2), "cliff": round(10 * float(b), 2)}
    return True


if __name__ == "__main__":
    import learn as _self   # re-import so pickles reference 'learn.EdgeModel', not '__main__.EdgeModel'
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    _self.train(ap.parse_args().input)
