#!/usr/bin/env python3
"""End-to-end promo finder.

  python run.py --input <episode.txt | folder> --out outputs [--llm] [--per-episode 2] [--pool 12]

Stages
  1 parse + propose all valid 20-60 s windows                         (code)
  2 rules score + LEARNED hook/cliff model (novelty 1)                  (code + sklearn model trained on the show's own edges)
  3 curiosity-gap analysis of the shortlist (novelty 2)                (LLM cold-viewer simulator, offline proxy w/o key)
  4 pairwise tournament over the finalists (novelty 3)                 (LLM judge both orders, offline majority vote w/o key)
  5 reports: promos.md / promos.json / promos.csv
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
import sys

import promo_finder as pf
import judge
import rank
import learn
import llm

W = {"hook": 0.35, "cliff": 0.30, "standalone": 0.20, "intensity": 0.15}


def combine(c, use_gap_for_standalone):
    """Blend rule scores with learned scores (novelty 1) and curiosity gap (novelty 2)."""
    s = dict(c.rules)
    if c.clf:
        s["hook"] = 0.5 * c.rules["hook"] + 0.5 * c.clf["hook"]
        s["cliff"] = 0.5 * c.rules["cliff"] + 0.5 * c.clf["cliff"]
    if use_gap_for_standalone and c.gap:
        s["standalone"] = 0.5 * c.rules["standalone"] + 0.5 * c.gap["gap"]
    c.scores = s
    L = c.end - c.start
    c.final = round(sum(W[k] * s[k] for k in W) - (0.0 if 25 <= L <= 50 else 0.4), 3)


def why_line(c):
    bits = []
    if c.scores["hook"] >= 6: bits.append("hook: " + (c.notes["hook"] or "strong opening"))
    if c.scores["cliff"] >= 6: bits.append("cliffhanger: " + (c.notes["cliff"] or "unresolved ending"))
    if c.gap and c.gap.get("gap", 0) >= 6: bits.append("high curiosity gap (clear alone, wants a next step)")
    if c.gap and c.gap.get("summary"): bits.append("stranger's view: " + c.gap["summary"])
    return "; ".join(bits) or "best available window; weakest on " + min(("hook", "cliff", "standalone"), key=lambda k: c.scores[k])


def load_episodes(path):
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".txt"))
    else:
        files = [path]
    eps = []
    for f in files:
        try:
            eps.append(pf.parse_episode(f))
        except ValueError as e:
            print("SKIP", e, file=sys.stderr)
    if not eps:
        sys.exit("no parsable episodes")
    return eps


def process(ep, use_llm, use_clf, per_ep, shortlist=4):
    cands = pf.propose(ep)
    for c in cands:
        c.rules = dict(c.scores)
    clf_ok = use_clf and learn.score_windows(ep, cands)
    for c in cands:
        combine(c, False)
    short = pf.select(cands, shortlist)
    if use_llm and llm.available() and short:
        with ThreadPoolExecutor(max_workers=4) as ex:
            gaps = list(ex.map(lambda c: judge.analyse(c, ep, use_llm), short))
        for c, g in zip(short, gaps):
            c.gap = g
            combine(c, True)
    else:
        for c in short:                                  # novelty 2 on the shortlist only (cheap)
            c.gap = judge.analyse(c, ep, use_llm)
            combine(c, True)
    short.sort(key=lambda c: -c.final)
    keep = []
    for c in short:
        if len(keep) < per_ep and not any(pf.overlaps(c, k) for k in keep):
            keep.append(c)
    return keep, len(cands), clf_ok


def fmt_row(r, c):
    return r


def report(ranked, out, meta):
    os.makedirs(out, exist_ok=True)
    rows = []
    for i, c in enumerate(ranked, 1):
        rows.append({
            "rank": i, "episode": c.episode,
            "segments": [{"start": pf.fmt(c.start), "end": pf.fmt(c.end)}],
            "duration_s": round(c.end - c.start, 1),
            "hook": c.hook_line, "cliffhanger": c.cliff_line, "why_it_works": why_line(c),
            "score": round(c.final, 2), "tournament_wins": c.wins,
            "sub_scores": {k: round(v, 1) for k, v in c.scores.items()},
            "rule_scores": {k: round(v, 1) for k, v in c.rules.items()},
            "learned_scores": c.clf, "curiosity_gap": c.gap,
        })
    json.dump({"meta": meta, "promos": rows}, open(os.path.join(out, "promos.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    with open(os.path.join(out, "promos.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "episode", "start", "end", "duration_s", "score", "wins", "hook", "cliffhanger", "why_it_works"])
        for r in rows:
            w.writerow([r["rank"], r["episode"], r["segments"][0]["start"], r["segments"][0]["end"], r["duration_s"],
                        r["score"], r["tournament_wins"], r["hook"], r["cliffhanger"], r["why_it_works"]])
    with open(os.path.join(out, "promos.md"), "w", encoding="utf-8") as f:
        f.write(f"# Promos, ranked\n\n`{json.dumps(meta)}`\n\n")
        f.write("| # | Ep | Start | End | Dur | Score | Wins | Hook | Cliffhanger |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['rank']} | {r['episode']} | {r['segments'][0]['start']} | {r['segments'][0]['end']} | {r['duration_s']}s | "
                    f"{r['score']} | {r['tournament_wins']} | {r['hook'][:80].replace('|', '/')} | {r['cliffhanger'][:80].replace('|', '/')} |\n")
        for c, r in zip(ranked, rows):
            f.write(f"\n## #{r['rank']} - Episode {c.episode}, {r['segments'][0]['start']} -> {r['segments'][0]['end']} "
                    f"({r['duration_s']}s, score {r['score']}, tournament wins {r['tournament_wins']})\n")
            f.write(f"- **Hook:** {r['hook']}\n- **Cliffhanger:** {r['cliffhanger']}\n- **Why it works:** {r['why_it_works']}\n")
            f.write(f"- Sub-scores {r['sub_scores']} | rules {r['rule_scores']} | learned {r['learned_scores']}\n")
            if c.gap:
                f.write(f"- Curiosity gap ({c.gap.get('source')}): {c.gap.get('gap')} - past confusions {c.gap.get('past_confusions')}, "
                        f"future questions {c.gap.get('future_questions')}\n")
            f.write("\n```\n" + c.transcript + "\n```\n")
    return rows


def find_promos(episodes, use_llm=False, use_clf=True, per_ep=2, pool=12):
    chosen, info = [], []
    for ep in episodes:
        keep, n, clf_ok = process(ep, use_llm, use_clf, per_ep)
        chosen += keep
        info.append((ep.number, len(ep.events), round(ep.duration, 1), n, len(keep), clf_ok))
    chosen.sort(key=lambda c: -c.final)
    finalists, rest = chosen[:pool], chosen[pool:]
    finalists, judge_kind = rank.tournament(finalists, use_llm)
    ranked = finalists + rest
    meta = {"episodes": len(episodes), "learned_model": bool(info and info[0][5]),
            "llm": llm.cfg()["model"] if (use_llm and llm.available()) else None,
            "tournament": judge_kind, "pool": len(finalists)}
    return ranked, meta, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="episode .txt or folder of episodes")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--llm", action="store_true", help="use the LLM (needs LLM_API_KEY)")
    ap.add_argument("--no-clf", action="store_true", help="disable the learned model (ablation)")
    ap.add_argument("--per-episode", type=int, default=2)
    ap.add_argument("--pool", type=int, default=12, help="finalists entering the pairwise tournament")
    a = ap.parse_args()
    if a.llm and not llm.available():
        print("WARNING: --llm given but LLM_API_KEY not set -> using offline fallbacks", file=sys.stderr)
    eps = load_episodes(a.input)
    ranked, meta, info = find_promos(eps, a.llm, not a.no_clf, a.per_episode, a.pool)
    for i in info:
        print("ep %2d: %3d events %6.1fs %4d windows -> %d kept (learned model: %s)" % i)
    report(ranked, a.out, meta)
    print(f"\n{len(ranked)} promos -> {a.out}/promos.md|json|csv   {meta}")


if __name__ == "__main__":
    main()
