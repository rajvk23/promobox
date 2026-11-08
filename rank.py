"""NOVELTY 3 - Pairwise tournament ranking ("which would you post first?").

Absolute 0-10 scores are noisy and un-calibrated across episodes. Humans (and LLMs) are more reliable at comparisons,
so the final ordering of the finalists comes from round-robin pairwise matches; rank = Copeland score (wins).

  * LLM judge : each pair is asked in BOTH orders (cancels position bias); a pair only counts as a win if both orders agree,
                else it is a draw (0.5 each).
  * offline   : majority-of-criteria vote (hook, cliffhanger, standalone/curiosity-gap): A beats B if it wins >= 2 of 3.
                This is a Condorcet-style rule, different from the weighted sum used to shortlist.
"""
from concurrent.futures import ThreadPoolExecutor
import itertools

import llm

PROMPT = """You are the editor-in-chief of a microdrama promo channel. Two clips from the same show are below.
Which one would you post FIRST to win new viewers? Judge: (1) hook in the first 3 seconds, (2) ends on an unresolved cliffhanger,
(3) makes sense to a stranger.

CLIP A (episode {ea}):
{ta}

CLIP B (episode {eb}):
{tb}

Return ONLY JSON: {{"winner": "A" or "B", "reason": "<=20 words"}}"""


def _crit(c):
    g = (c.gap or {}).get("gap", c.scores["standalone"])
    return (c.scores["hook"] if not c.clf else 0.5 * (c.scores["hook"] + c.clf["hook"]),
            c.scores["cliff"] if not c.clf else 0.5 * (c.scores["cliff"] + c.clf["cliff"]),
            g)


def _offline(a, b):
    ca, cb = _crit(a), _crit(b)
    wa = sum(x > y + 0.05 for x, y in zip(ca, cb)); wb = sum(y > x + 0.05 for x, y in zip(ca, cb))
    return 1.0 if wa > wb else 0.0 if wb > wa else 0.5


def _llm_pair(a, b):
    try:
        r1 = llm.chat_json(PROMPT.format(ea=a.episode, ta=a.transcript, eb=b.episode, tb=b.transcript))["winner"]
        r2 = llm.chat_json(PROMPT.format(ea=b.episode, ta=b.transcript, eb=a.episode, tb=a.transcript))["winner"]
    except Exception:
        return _offline(a, b)
    a_first, a_second = r1 == "A", r2 == "B"
    return 1.0 if a_first and a_second else 0.0 if (not a_first and not a_second) else 0.5


def tournament(cands, use_llm):
    """Round-robin; sets c.wins and returns the cands sorted best-first. Ties broken by weighted score (c.final)."""
    is_llm = bool(use_llm and llm.available())
    judge = _llm_pair if is_llm else _offline
    for c in cands:
        c.wins = 0.0
    pairs = list(itertools.combinations(cands, 2))
    if is_llm and pairs:
        with ThreadPoolExecutor(max_workers=8) as ex:
            scores = list(ex.map(lambda ab: judge(ab[0], ab[1]), pairs))
    else:
        scores = [judge(a, b) for a, b in pairs]
    for (a, b), s in zip(pairs, scores):
        a.wins += s; b.wins += 1 - s
    return sorted(cands, key=lambda c: (-c.wins, -c.final)), ("llm" if is_llm else "offline-majority")
