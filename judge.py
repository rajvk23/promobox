"""NOVELTY 2 - Curiosity-gap score ("does it make sense on its own?" made measurable).

A first-time viewer sees ONLY the clip. We ask what they are CONFUSED about from the past (bad: the clip needs
backstory) and what they are DYING TO KNOW about the future (good: cliffhanger pull).

    gap = clamp( 10 * (  min(future_questions, 4)/4  -  0.5 * min(past_confusions, 4)/4 ) + 3 )   in [0, 10]
    -> few backward-looking confusions + several forward-looking questions = high score.

Two implementations with identical output shape:
  * LLM   : cold-viewer simulator (accurate, needs API key)
  * proxy : offline heuristic (names/titles mentioned but never shown speaking, references to unseen past events
            = past confusion;  questions, threats, reveals near the end = future pull).  Used when no key is set.
"""
import re

import llm
import promo_finder as pf

PROMPT = """You are a first-time viewer scrolling a vertical-video app. You have NEVER seen this show.
You are shown ONLY this clip (a timestamped screenplay excerpt). Do not use outside knowledge.

CLIP:
{transcript}

1. List ONLY plot-breaking confusions (past_confusions): things that make the clip HARD TO FOLLOW because it refers to prior events, unintroduced names, or missing context that a stranger cannot infer. If the situation is self-explanatory (e.g. a bride at a wedding, a guard arresting someone), return []. Do NOT list normal dramatic curiosity as past confusion.
2. List what you desperately want to know NEXT (future_questions) - genuine unresolved stakes/cliffhangers at the end of the clip (0 to 4 items).
3. Rate `standalone_clarity` (0-10): how easily a stranger understands what is happening right now.
4. Rate `cliffhanger_pull` (0-10): how urgently a stranger needs to watch the next second after the clip cuts.
5. Summarise the clip in one sentence as a stranger would.

Return ONLY JSON:
{{"past_confusions": ["..."], "future_questions": ["..."], "standalone_clarity": 0-10, "cliffhanger_pull": 0-10, "summary": "..."}}"""

BACKREF = ["again", "still", "last night", "yesterday", "before", "as promised", "you promised", "remember", "earlier",
           "once more", "already", "back then", "i told you", "that day", "what happened", "after all"]


def gap_from_counts(past, future):
    p, f = min(past, 4) / 4.0, min(future, 4) / 4.0
    return round(pf.clamp(10 * (f - 0.5 * p) + 3), 2)


def proxy(c, ep):
    """Offline estimate (no LLM)."""
    evs = [e for e in ep.events if e.end > c.start and e.start < c.end and e.kind in ("DIALOGUE", "ACTION")]
    speakers = {e.speaker.lower() for e in evs if e.kind == "DIALOGUE"}
    dtext = " ".join(e.text for e in evs if e.kind == "DIALOGUE")
    # capitalised mid-sentence names/titles mentioned but never speaking in the clip = unseen people
    mentioned = {m for m in re.findall(r"(?<![.?!\"] )(?<!^)\b([A-Z][a-z]{2,})\b", dtext)}
    mentioned -= {"The", "You", "Your", "What", "How", "Why", "Who", "But", "And", "Not", "Now", "Then", "Yes"}
    unseen = [m for m in mentioned if not any(m.lower() in s for s in speakers)]
    back = pf.kw_hits(dtext.lower(), BACKREF)
    past = min(4, len(unseen) + back)
    tail = [e for e in evs if e.end > c.end - 10]
    ttxt = " ".join(e.text for e in tail)
    future = min(4, ttxt.count("?") + (1 if pf.kw_hits(ttxt, pf.HOOK_WORDS) else 0)
                 + (1 if pf.kw_hits(ttxt, pf.REVEAL_PHRASES) else 0) + (1 if c.end >= ep.duration - 1.5 else 0))
    return {"past_confusions": [f"unseen: {u}" for u in unseen[:past]] + (["back-references"] if back else []),
            "future_questions": ["open question/threat/reveal at end"] * future,
            "summary": "", "source": "proxy", "gap": gap_from_counts(past, future)}


def analyse(c, ep, use_llm):
    if use_llm and llm.available():
        try:
            j = llm.chat_json(PROMPT.format(transcript=c.transcript))
            past, fut = len(j.get("past_confusions", [])), len(j.get("future_questions", []))
            count_gap = gap_from_counts(past, fut)
            if "standalone_clarity" in j and "cliffhanger_pull" in j:
                rated_gap = 0.5 * float(j["standalone_clarity"]) + 0.5 * float(j["cliffhanger_pull"]) - 0.8 * min(past, 4)
                gap_val = round(pf.clamp(0.5 * count_gap + 0.5 * rated_gap), 2)
            else:
                gap_val = count_gap
            j.update({"source": llm.cfg()["model"], "gap": gap_val})
            return j
        except Exception as e:
            r = proxy(c, ep); r["note"] = f"LLM failed, used proxy: {str(e)[:80]}"
            return r
    return proxy(c, ep)
