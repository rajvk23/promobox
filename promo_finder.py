#!/usr/bin/env python3
"""Promo finder for microdrama screenplays.

Pipeline
  1. parse   : screenplay .txt -> timed events (dialogue / action / sound / camera)   [code]
  2. propose : every 20-60 s window that starts and ends on an event boundary         [code]
  3. score   : hook (first 3 s) / cliffhanger (last seconds) / standalone / intensity [rules]
  4. (opt.)  : LLM re-scores the top candidates and its scores are blended in         [LLM, swappable]
  5. select  : non-overlapping top-N per episode, global ranking, reports             [code]

Usage
  python promo_finder.py --input "<folder with episode-*.txt>" --out outputs
  python promo_finder.py --input ... --out outputs --llm      # needs LLM_API_KEY etc. (see llm_scorer.py)
"""
import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict

MIN_LEN, MAX_LEN = 20.0, 60.0

# ----------------------------------------------------------------------------- parsing
TS = r"(\d+):(\d+(?:\.\d+)?)"
RE_TIMES = re.compile(r"Start:\s*\(?" + TS + r"\)?\s*\|\s*End:\s*\(?" + TS + r"\)?")
RE_HEAD = re.compile(r"^\*\*(.+?):\*\*\s*$")
RE_SCENE_START = re.compile(r"Scene start:\s*" + TS)
RE_SCENE_HEAD = re.compile(r"Scene heading:\s*(.+)")


def secs(m, s):
    return int(m) * 60 + float(s)


@dataclass
class Event:
    kind: str          # DIALOGUE | ACTION | CAMERA | SOUND
    speaker: str
    emotion: str
    start: float
    end: float
    text: str
    scene: int = 0


@dataclass
class Episode:
    number: int
    path: str
    logline: str
    events: list = field(default_factory=list)
    scene_starts: dict = field(default_factory=dict)   # scene idx -> (start, heading)
    duration: float = 0.0


def read_text(path):
    raw = open(path, "rb").read()
    try:
        t = raw.decode("utf-8")
    except UnicodeDecodeError:
        t = raw.decode("latin-1")
    # repair mojibake (utf-8 read as cp1252) if present
    if "â€" in t or "Ã" in t:
        try:
            t = t.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            pass
    return t


def parse_episode(path):
    return parse_text(read_text(path), os.path.basename(path), path)


def parse_text(text, name="episode", path=""):
    """Parse screenplay text (also used for uploaded/pasted episodes)."""
    lines = text.splitlines()
    m = re.search(r"episode-(\d+)", name, re.I) or re.search(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", text, re.I) \
        or re.search(r"(\d+)", name)
    num = int(m.group(1)) if m else 1
    ep = Episode(num, path or name, "")
    scene, cur = 0, None
    pending_scene_head = None
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("EPISODE LOGLINE:"):
            ep.logline = ln.split(":", 1)[1].strip()
        elif ln.startswith("SCENE "):
            scene += 1
        elif RE_SCENE_HEAD.match(ln):
            pending_scene_head = RE_SCENE_HEAD.match(ln).group(1)
        elif RE_SCENE_START.match(ln):
            m = RE_SCENE_START.match(ln)
            ep.scene_starts[scene] = (secs(*m.groups()), pending_scene_head or "")
        else:
            h = RE_HEAD.match(ln)
            if h and i + 1 < len(lines):
                tm = RE_TIMES.search(lines[i + 1])
                if tm:
                    g = tm.groups()
                    st, en = secs(g[0], g[1]), secs(g[2], g[3])
                    label = h.group(1)
                    emo = ""
                    name = label.split("(")[0].strip() or label
                    groups = re.findall(r"\(([^()]*)\)", label)   # handles "JULIANA (V.O.) (Gentle, Surprised)"
                    emo = ", ".join(g.strip() for g in groups)
                    # body = following lines until blank
                    j, body = i + 2, []
                    while j < len(lines) and lines[j].strip():
                        body.append(lines[j].strip())
                        j += 1
                    text = " ".join(body).strip()
                    if name in ("CAMERA",):
                        kind = "CAMERA"
                    elif name == "ACTION":
                        kind = "ACTION"
                    elif name in ("SOUND FX", "SOUND", "MUSIC"):
                        kind = "SOUND"
                    else:
                        kind = "DIALOGUE"
                        text = text.strip('"')
                    ep.events.append(Event(kind, name, emo, st, en, text, scene))
                    ep.duration = max(ep.duration, en)
                    i = j
                    continue
        i += 1
    ep.events.sort(key=lambda e: (e.start, e.end))
    if len(ep.events) < 5:
        raise ValueError(f"{name}: could not find timed events (expected blocks like '**NAME (emotion):**' "
                         f"followed by 'Start: (mm:ss.ss) | End: (mm:ss.ss)').")
    return ep


# ----------------------------------------------------------------------------- lexicons
HOOK_WORDS = {
    # violence / shock
    "slap", "slaps", "slapped", "strikes", "struck", "punch", "blood", "bleeding", "knife", "dagger", "sword", "gun",
    "kill", "killed", "murder", "murdered", "dead", "death", "die", "poison", "poisoned", "scream", "screams",
    "shatters", "slams", "grabs", "drags", "throws", "kneels", "collapses", "faints",
    # reveals / stakes
    "secret", "truth", "lie", "lied", "liar", "betray", "betrayed", "betrayal", "pregnant", "baby", "child", "heir",
    "marry", "marriage", "wife", "husband", "divorce", "bastard", "traitor", "spy", "imposter", "impostor",
    "duke", "duchess", "king", "queen", "prince", "fortune", "inheritance", "scandal", "affair", "forbidden",
    "never", "dare", "how dare", "get out", "leave", "banish", "banished", "arrest", "guards", "execute", "hang",
    "mine", "kiss", "kisses", "naked", "shirtless", "bed", "burn", "burns", "fire", "escape", "run",
    "who are you", "what did you", "you will pay", "regret", "mercy", "beg", "begs", "crawl",
}
REVEAL_PHRASES = [
    "you are", "you're not", "it was you", "it was me", "i know what", "i know who", "i saw", "i am the", "i'm the",
    "was never", "have been", "all along", "the whole time", "not what", "what have you", "what did you",
    "who are you", "who sent", "what is this", "impossible", "can't be", "cannot be",
]
CLIFF_SOUND = ["cliffhanger", "cut to black", "ominous", "sting", "gasp", "drone", "swell", "silence", "heartbeat",
               "freeze", "abrupt", "slam", "shatter", "scream"]
CLIFF_ACTION = ["cut to black", "freezes", "stunned", "horrified", "gasp", "shock", "sees", "discovers", "realizes",
                "terror", "stares", "opens the door", "door", "walks in", "witness", "enters", "appears", "reveals",
                "behind", "approaches", "blade", "lethal", "unaware", "smirk", "sinister", "whisper"]
RESOLUTION_WORDS = ["thank you", "thanks", "goodbye", "good night", "farewell", "very well", "as you wish",
                    "i understand", "i forgive", "all is well", "at last", "together", "finally"]
INTENSE_EMO = ["shout", "scream", "furious", "rage", "fury", "outrag", "gasp", "shock", "terrif", "panic",
               "cold", "menac", "threat", "vicious", "desperate", "sneer", "sinister", "trembling", "fierce",
               "defiant", "accus", "stunned", "horrified", "triumph", "gloat", "sharp", "seething", "chilling"]
WEAK_OPENERS = {"yes", "no", "yeah", "okay", "ok", "well", "and", "but", "so", "then", "thank", "thanks", "indeed",
                "i see", "very", "fine", "of course", "alright", "right", "he", "she", "they", "it", "that", "this"}

WORD = re.compile(r"[a-z']+")


def words(t):
    return WORD.findall(t.lower())


def kw_hits(text, vocab):
    t = " " + text.lower() + " "
    return sum(1 for w in vocab if (" " + w + " ") in t or (" " + w + ",") in t or w + "." in t or w + "!" in t
               or w + "?" in t or (" " in w and w in t))


def emo_score(emo):
    e = emo.lower()
    return sum(1 for w in INTENSE_EMO if w in e)


def clamp(x, lo=0.0, hi=10.0):
    return max(lo, min(hi, x))


# ----------------------------------------------------------------------------- scoring
def story_events(evs):
    return [e for e in evs if e.kind in ("DIALOGUE", "ACTION") or (e.kind == "SOUND")]


def score_hook(window_events, start):
    """Hook = what happens in the first 3 seconds of the clip."""
    first = [e for e in window_events if e.kind != "CAMERA" and e.start < start + 3.0 and e.end > start]
    if not first:
        return 0.0, "no dialogue/action in first 3s"
    s, why = 2.0, []
    for e in first:
        txt = e.text
        h = kw_hits(txt, HOOK_WORDS) + kw_hits(txt, REVEAL_PHRASES)
        if e.kind == "DIALOGUE":
            if "!" in txt:
                s += 1.5; why.append("exclamation")
            if "?" in txt and len(words(txt)) >= 3:
                s += 1.0; why.append("question")
            if h:
                s += min(3.0, 1.5 * h); why.append("charged words")
            s += min(2.0, emo_score(e.emotion) * 0.8)
            first_w = " ".join(words(txt)[:2])
            if words(txt) and (words(txt)[0] in WEAK_OPENERS or first_w in WEAK_OPENERS):
                s -= 1.5; why.append("weak opener")
            if len(words(txt)) < 2:
                s -= 1.0
        elif e.kind == "ACTION":
            if h:
                s += min(3.0, 1.5 * h); why.append("striking action")
        elif e.kind == "SOUND":
            if kw_hits(txt, CLIFF_SOUND):
                s += 1.0
    return clamp(s), ", ".join(dict.fromkeys(why)) or "mild opening"


def score_cliff(window_events, end, ep):
    """Cliffhanger = the last ~6 seconds leave something unresolved."""
    last = [e for e in window_events if e.end > end - 6.0]
    if not last:
        return 0.0, "nothing at end"
    s, why = 1.0, []
    at_episode_end = end >= ep.duration - 1.5
    if at_episode_end:
        s += 3.0; why.append("ends on episode's own cliffhanger")
    for e in last:
        txt = e.text
        low = txt.lower()
        if e.kind == "SOUND" and kw_hits(low, CLIFF_SOUND):
            s += 1.5; why.append("sound sting")
        elif e.kind == "ACTION":
            c = kw_hits(low, CLIFF_ACTION)
            if c:
                s += min(2.5, 1.0 * c); why.append("visual reveal/shock")
        elif e.kind == "DIALOGUE":
            if "?" in txt:
                s += 1.5; why.append("ends on question")
            if txt.rstrip().endswith("...") or txt.rstrip().endswith("—"):
                s += 1.0; why.append("trailing line")
            c = kw_hits(txt, HOOK_WORDS) + kw_hits(txt, REVEAL_PHRASES)
            if c:
                s += min(2.0, 1.0 * c); why.append("threat/reveal line")
            s += min(1.5, emo_score(e.emotion) * 0.6)
            if kw_hits(low, RESOLUTION_WORDS):
                s -= 2.0; why.append("sounds resolved")
    # last story event matters most
    story = [e for e in window_events if e.kind in ("DIALOGUE", "ACTION")]
    if story:
        le = story[-1]
        if le.kind == "DIALOGUE" and "." in le.text and "?" not in le.text and "!" not in le.text \
                and not at_episode_end and not kw_hits(le.text, HOOK_WORDS):
            s -= 1.0; why.append("flat last line")
    return clamp(s), ", ".join(dict.fromkeys(why)) or "no clear cliffhanger"


def score_standalone(window_events, start, end, ep):
    """Does it make sense to a first-time viewer? Rules only; the LLM judges this better."""
    s, why = 6.0, []
    speakers = {e.speaker for e in window_events if e.kind == "DIALOGUE"}
    dlg = [e for e in window_events if e.kind == "DIALOGUE"]
    # starts at the top of a scene -> setting is established inside the clip
    for idx, (st, head) in ep.scene_starts.items():
        if abs(st - start) < 1.0:
            s += 1.5; why.append("starts at scene start"); break
    # does it cross a scene cut? (more cuts = harder to follow)
    scenes = {e.scene for e in window_events}
    if len(scenes) > 2:
        s -= 1.0 * (len(scenes) - 2); why.append("many scene changes")
    if len(speakers) > 4:
        s -= 1.0; why.append("crowded cast")
    if len(speakers) == 0:
        s -= 3.0; why.append("no dialogue")
    if len(speakers) == 1 and len(dlg) <= 1:
        s -= 1.0
    # first dialogue should not be a back-channel reply
    if dlg:
        fw = words(dlg[0].text)
        if fw and fw[0] in WEAK_OPENERS:
            s -= 1.0; why.append("opens mid-conversation")
    # earlier episodes need less backstory
    if ep.number <= 10:
        s += 1.0; why.append("early episode (less backstory)")
    elif ep.number > 35:
        s -= 1.0; why.append("late episode (heavy backstory)")
    # a clip made only of talk with no action is dull
    if not any(e.kind == "ACTION" for e in window_events):
        s -= 0.5
    return clamp(s), ", ".join(dict.fromkeys(why)) or "ok"


def score_intensity(window_events):
    txt = " ".join(e.text for e in window_events if e.kind in ("DIALOGUE", "ACTION"))
    h = kw_hits(txt, HOOK_WORDS) + kw_hits(txt, REVEAL_PHRASES)
    emo = sum(emo_score(e.emotion) for e in window_events if e.kind == "DIALOGUE")
    n = max(1, len([e for e in window_events if e.kind in ("DIALOGUE", "ACTION")]))
    return clamp(10.0 * (h + 0.7 * emo) / (n * 1.2))


WEIGHTS = {"hook": 0.35, "cliff": 0.30, "standalone": 0.20, "intensity": 0.15}


def total(sc):
    return sum(WEIGHTS[k] * sc[k] for k in WEIGHTS)


# ----------------------------------------------------------------------------- candidates
@dataclass
class Candidate:
    episode: int
    start: float
    end: float
    scores: dict
    notes: dict
    final: float = 0.0
    transcript: str = ""
    hook_line: str = ""
    cliff_line: str = ""
    llm: dict = None
    clf: dict = None       # learned hook/cliff scores (novelty 1)
    gap: dict = None       # curiosity-gap analysis (novelty 2)
    wins: float = 0.0      # pairwise tournament wins (novelty 3)


def fmt(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def event_line(e):
    if e.kind == "DIALOGUE":
        return f'{e.speaker}{(" (" + e.emotion + ")") if e.emotion else ""}: "{e.text}"'
    if e.kind == "SOUND":
        return f"[SOUND] {e.text}"
    if e.kind == "CAMERA":
        return f"[CAMERA] {e.text}"
    return f"[ACTION] {e.text}"


def transcript_of(evs, with_time=True, with_camera=False):
    out = []
    for e in evs:
        if e.kind == "CAMERA" and not with_camera:
            continue
        out.append((f"[{fmt(e.start)}] " if with_time else "") + event_line(e))
    return "\n".join(out)


def propose(ep):
    """Windows begin at the start of a dialogue/action event and end at the end of an event."""
    evs = ep.events
    starts = sorted({e.start for e in evs if e.kind in ("DIALOGUE", "ACTION")})
    ends = sorted({e.end for e in evs if e.kind in ("DIALOGUE", "ACTION", "SOUND")} | {ep.duration})
    cands = []
    for st in starts:
        for en in ends:
            L = en - st
            if L < MIN_LEN:
                continue
            if L > MAX_LEN:
                break
            w = [e for e in evs if e.end > st + 0.01 and e.start < en - 0.01]
            # skip windows whose first/last event is cut in half
            if any(e.start < st - 0.01 and e.kind == "DIALOGUE" for e in w):
                continue
            if any(e.end > en + 0.01 and e.kind == "DIALOGUE" for e in w):
                continue
            sc = {}
            sc["hook"], hn = score_hook(w, st)
            sc["cliff"], cn = score_cliff(w, en, ep)
            sc["standalone"], sn = score_standalone(w, st, en, ep)
            sc["intensity"] = score_intensity(w)
            c = Candidate(ep.number, st, en, sc, {"hook": hn, "cliff": cn, "standalone": sn})
            # duration preference: 25-45 s is the sweet spot for short-form
            dur_pen = 0.0 if 25 <= L <= 50 else 0.4
            c.final = round(total(sc) - dur_pen, 3)
            c.transcript = transcript_of(w)
            d0 = [e for e in w if e.kind in ("DIALOGUE", "ACTION") and e.start < st + 3.0 + 1e-6]
            c.hook_line = event_line(d0[0]) if d0 else ""
            sl = [e for e in w if e.kind in ("DIALOGUE", "ACTION")]
            c.cliff_line = event_line(sl[-1]) if sl else ""
            cands.append(c)
    return cands


def overlaps(a, b):
    return a.episode == b.episode and a.start < b.end and b.start < a.end


def select(cands, per_ep):
    cands = sorted(cands, key=lambda c: -c.final)
    chosen = []
    for c in cands:
        if sum(1 for x in chosen if x.episode == c.episode) >= per_ep:
            continue
        if any(overlaps(c, x) for x in chosen):
            continue
        chosen.append(c)
    return chosen


# ----------------------------------------------------------------------------- reports
def why_line(c):
    bits = []
    if c.scores["hook"] >= 6: bits.append("strong opening (" + c.notes["hook"] + ")")
    if c.scores["cliff"] >= 6: bits.append("unresolved ending (" + c.notes["cliff"] + ")")
    if c.scores["standalone"] >= 7: bits.append("easy to follow cold")
    if not bits:
        bits.append("best available window in this episode; weak on " +
                    min(("hook", "cliff", "standalone"), key=lambda k: c.scores[k]))
    return "; ".join(bits)


def write_reports(chosen, out, use_llm):
    os.makedirs(out, exist_ok=True)
    ranked = sorted(chosen, key=lambda c: -c.final)
    rows = []
    for r, c in enumerate(ranked, 1):
        rows.append({
            "rank": r, "episode": c.episode,
            "segments": [{"start": fmt(c.start), "end": fmt(c.end)}],
            "duration_s": round(c.end - c.start, 1),
            "hook": c.hook_line, "cliffhanger": c.cliff_line, "why_it_works": why_line(c),
            "score": round(c.final, 2), "sub_scores": {k: round(v, 1) for k, v in c.scores.items()},
            "llm": c.llm, "scorer": "rules+llm" if c.llm else "rules",
        })
    json.dump(rows, open(os.path.join(out, "promos.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(os.path.join(out, "promos.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "episode", "start", "end", "duration_s", "score", "hook", "cliffhanger", "why_it_works"])
        for r in rows:
            w.writerow([r["rank"], r["episode"], r["segments"][0]["start"], r["segments"][0]["end"],
                        r["duration_s"], r["score"], r["hook"], r["cliffhanger"], r["why_it_works"]])
    with open(os.path.join(out, "promos.md"), "w", encoding="utf-8") as f:
        f.write(f"# Promo candidates ({len(rows)}), ranked  (scorer: {'rules + LLM' if use_llm else 'rules only'})\n\n")
        f.write("| # | Ep | Start | End | Dur | Score | Hook | Cliffhanger |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['rank']} | {r['episode']} | {r['segments'][0]['start']} | {r['segments'][0]['end']} | "
                    f"{r['duration_s']}s | {r['score']} | {r['hook'][:90].replace('|', '/')} | "
                    f"{r['cliffhanger'][:90].replace('|', '/')} |\n")
        f.write("\n---\n")
        for c, r in zip(ranked, rows):
            f.write(f"\n## #{r['rank']} — Episode {c.episode}, {r['segments'][0]['start']} → {r['segments'][0]['end']} "
                    f"({r['duration_s']}s, score {r['score']})\n")
            f.write(f"- **Hook:** {r['hook']}\n- **Cliffhanger:** {r['cliffhanger']}\n- **Why it works:** {r['why_it_works']}\n")
            f.write(f"- Sub-scores: {r['sub_scores']}\n")
            if c.llm:
                f.write(f"- LLM: {c.llm}\n")
            f.write("\n```\n" + c.transcript + "\n```\n")
    return ranked


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder with episode-*.txt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--per-episode", type=int, default=2, help="max promos kept per episode")
    ap.add_argument("--top", type=int, default=0, help="keep only the N best overall (0 = all kept)")
    ap.add_argument("--llm", action="store_true", help="re-score top candidates with an LLM (see llm_scorer.py)")
    ap.add_argument("--llm-pool", type=int, default=4, help="candidates per episode sent to the LLM")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith(".txt"))
    if not files:
        sys.exit("no .txt files found")
    all_chosen, stats = [], []
    for fn in files:
        ep = parse_episode(os.path.join(a.input, fn))
        cands = propose(ep)
        pool = select(cands, max(a.per_episode, a.llm_pool if a.llm else a.per_episode))
        if a.llm:
            import llm_scorer
            llm_scorer.rescore(ep, pool)
            pool.sort(key=lambda c: -c.final)
        chosen = []
        for c in pool:
            if len(chosen) < a.per_episode and not any(overlaps(c, x) for x in chosen):
                chosen.append(c)
        all_chosen += chosen
        stats.append((ep.number, len(ep.events), round(ep.duration, 1), len(cands), len(chosen)))
        print(f"ep {ep.number:>2}: {len(ep.events):>3} events, {ep.duration:>6.1f}s, {len(cands):>4} windows -> {len(chosen)} kept")
    if a.top:
        all_chosen = sorted(all_chosen, key=lambda c: -c.final)[:a.top]
    ranked = write_reports(all_chosen, a.out, a.llm)
    print(f"\n{len(ranked)} promos written to {a.out}/ (promos.md, promos.json, promos.csv)")


if __name__ == "__main__":
    main()
