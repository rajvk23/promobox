"""Quality checking.

  python eval.py sanity  --input <folder>             # hard constraints on every output promo (no labels needed)
  python eval.py make-gold --input <folder>           # writes gold/gold_template.csv with ~24 clips to label (1=good promo, 0=bad)
  python eval.py score                                # after you fill gold/gold.csv: AUC / precision@k per variant

Variants compared on the same clips: rules | rules+learned(novelty 1) | full (adds curiosity gap, novelty 2).
"""
import argparse
import csv
import os
import random
import sys

import numpy as np

import promo_finder as pf
import run as R
import judge

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold")


def all_candidates(eps, use_llm=False):
    """Returns candidates for the per-episode shortlist, each carrying variant scores."""
    out = []
    for ep in eps:
        keep, _, _ = R.process(ep, use_llm, True, per_ep=4, shortlist=4)
        for c in keep:
            c.ep = ep
            rules_only = sum(R.W[k] * c.rules[k] for k in R.W)
            c.variants = {"rules": round(rules_only, 3)}
            out.append(c)
    return out


def variant_scores(c, ep):
    # rules+learned: blend hook/cliff with learned model, standalone from rules only
    s = dict(c.rules)
    if c.clf:
        s["hook"] = .5 * c.rules["hook"] + .5 * c.clf["hook"]; s["cliff"] = .5 * c.rules["cliff"] + .5 * c.clf["cliff"]
    L = c.end - c.start
    pen = 0.0 if 25 <= L <= 50 else 0.4
    c.variants["rules"] = round(sum(R.W[k] * c.rules[k] for k in R.W) - pen, 3)
    c.variants["rules+learned"] = round(sum(R.W[k] * s[k] for k in R.W) - pen, 3)
    s_proxy = dict(s)
    s_proxy["standalone"] = .5 * c.rules["standalone"] + .5 * judge.proxy(c, ep)["gap"]
    c.variants["full_offline"] = round(sum(R.W[k] * s_proxy[k] for k in R.W) - pen, 3)
    c.variants["full_llm"] = round(c.final, 3)


def sanity(eps):
    ranked, meta, _ = R.find_promos(eps, False, True, 2, 12)
    bad = 0
    byep = {e.number: e for e in eps}
    for c in ranked:
        ep = byep[c.episode]
        L = c.end - c.start
        evs = [e for e in ep.events if e.kind == "DIALOGUE" and e.end > c.start and e.start < c.end]
        cut = [e for e in evs if e.start < c.start - .01 or e.end > c.end + .01]
        probs = []
        if not 20 <= L <= 60: probs.append(f"length {L:.1f}")
        if cut: probs.append("cuts a line in half")
        if c.end > ep.duration + .01 or c.start < 0: probs.append("outside episode")
        if probs:
            bad += 1; print("BAD", c.episode, pf.fmt(c.start), pf.fmt(c.end), probs)
    print(f"sanity: {len(ranked)} promos, {bad} violating hard constraints")


def make_gold(eps, n=24, seed=1):
    import llm
    cands = all_candidates(eps, use_llm=llm.available())
    for c in cands:
        variant_scores(c, c.ep)
    cands.sort(key=lambda c: -c.final)
    rnd = random.Random(seed)
    top, bottom = cands[:8], cands[-8:]
    mid = rnd.sample(cands[8:-8], n - 16)
    os.makedirs(GOLD, exist_ok=True)
    rows = top + mid + bottom
    rnd.shuffle(rows)                       # so you label blind to the model's rank
    with open(os.path.join(GOLD, "gold_template.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["episode", "start", "end", "human_label_1good_0bad", "rules", "rules+learned", "full_offline", "full_llm", "transcript"])
        for c in rows:
            w.writerow([c.episode, pf.fmt(c.start), pf.fmt(c.end), "", c.variants["rules"], c.variants["rules+learned"],
                        c.variants["full_offline"], c.variants["full_llm"], c.transcript.replace("\n", " // ")])
    print(f"wrote {len(rows)} clips to gold/gold_template.csv. Fill column D (1 good / 0 bad), save as gold/gold.csv")


def auc(y, s):
    pos = [a for a, b in zip(s, y) if b == 1]; neg = [a for a, b in zip(s, y) if b == 0]
    if not pos or not neg: return float("nan")
    return float(np.mean([(p > n) + .5 * (p == n) for p in pos for n in neg]))


def score():
    p = os.path.join(GOLD, "gold.csv")
    if not os.path.exists(p):
        sys.exit("create gold/gold.csv from gold_template.csv first")
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8-sig")) if r["human_label_1good_0bad"].strip() in ("0", "1")]
    y = [int(r["human_label_1good_0bad"]) for r in rows]
    print(f"{len(rows)} labelled clips, {sum(y)} good")
    cols = [v for v in ("rules", "rules+learned", "full_offline", "full_llm", "full") if rows and v in rows[0]]
    for v in cols:
        s = [float(r[v]) for r in rows]
        k = max(1, len(rows) // 4)
        top = np.argsort(s)[::-1][:k]
        print(f"{v:14s} AUC={auc(y, s):.3f}  precision@{k}={np.mean([y[i] for i in top]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sanity", "make-gold", "score"])
    ap.add_argument("--input")
    a = ap.parse_args()
    if a.cmd == "score":
        score()
    else:
        eps = R.load_episodes(a.input)
        (sanity if a.cmd == "sanity" else make_gold)(eps)
