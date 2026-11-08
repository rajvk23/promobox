# Microdrama Promo Finder (Kuku — Problem 1)

Reads timestamped microdrama screenplays (`.txt`), proposes boundary-safe 20–60 s clips, scores and ranks them through a 4-stage pipeline, and outputs exact timestamps, hook, cliffhanger, and a one-line editorial rationale.

## Quick Start (works on any fresh episode or folder)
```bash
pip install -r requirements.txt

# 1. Fast offline run (Rules + Self-Supervised ML + Offline Tournament) — ~3 s for 50 episodes, $0 cost:
python run.py --input "<episode.txt or folder>" --out outputs_offline

# 2. Full LLM run (adds Cold-Viewer Curiosity-Gap Simulator + Pairwise LLM Tournament):
set LLM_API_KEY=your_key_here
python run.py --input "<episode.txt or folder>" --out outputs --llm

# 3. Interactive Web App (upload fresh episodes in browser, inspect transcripts, download JSON):
streamlit run app.py
```

### Swapping the Model (zero code changes)
`llm.py` supports both Google's native `generateContent` REST endpoint and any OpenAI-compatible `/chat/completions` endpoint via environment variables:
- `LLM_API_KEY`: API key (if unset, the pipeline automatically falls back to offline mode without crashing)
- `LLM_MODEL`: defaults to `gemini-3.1-flash-lite` (tested with `gemini-3.5-flash`, `gemini-3-flash-preview`, `gpt-4o-mini`, etc.)
- `LLM_BASE_URL`: defaults to `https://generativelanguage.googleapis.com/v1beta/openai` (auto-routes to native Gemini REST when `generativelanguage.googleapis.com` is in the URL, or standard OpenAI chat format for OpenAI / OpenRouter / Groq / vLLM).

---

## Architecture: What is Code vs. ML vs. LLM vs. Human

| Stage | Responsibility | Engine | Cost & Latency (50 eps) |
|---|---|---|---|
| **1. Parse & Propose** | Parse timestamps, speakers, emotions, actions; generate all 20–60 s windows aligned to event boundaries (never cuts a line in half) | Deterministic Python (`promo_finder.py`) | ~0.5 s, $0 |
| **2a. Structural Rules** | Score Hook (first 3 s), Cliffhanger (last 6 s), Standalone clarity, and Dramatic Intensity | Lexicon + punctuation + emotion-tag rules (`promo_finder.py`) | ~1.0 s, $0 |
| **2b. Novelty 1: Self-Supervised Edge Classifier** | Predict hook & cliffhanger probability trained on writer-chosen episode openings & endings | `scikit-learn` Logistic Regression (`learn.py`, `models/*.pkl`) | ~1.5 s, $0 |
| **3. Novelty 2: Cold-Viewer Curiosity Gap** | Simulate a first-time viewer on each episode's top 4 shortlist; measure `future_questions` vs. `past_confusions` | LLM (`judge.py`) with offline heuristic fallback | 200 calls (~25 s parallel, cached), ~$0.015 |
| **4. Novelty 3: Pairwise Tournament** | Round-robin head-to-head comparison ("which would you post first?") across top 12 finalists in **both orders** (cancels position bias) | LLM (`rank.py`) with Condorcet majority-vote fallback | 132 calls (~15 s parallel, cached), ~$0.010 |
| **5. Editorial Selection** | Pick final clip, add music/captions, verify visual execution | **Human Editor** | Reviews top 10–15 instead of watching 50 episodes |

---

## The Three Novelties

1. **Self-Supervised Hook & Cliffhanger Classifier (`learn.py`)**
   - *Insight:* Every episode in a microdrama is written to open on a hook (first 3 s) and end on a cliffhanger (last 6 s). That gives 50 positive examples per class with zero manual labelling.
   - *Generalisation:* We strip stage directions that leak the edge (`cut to black`, `cliffhanger`), mask character names, match negative samples by event type, filter out tender/resolved endings using rule medians, and ship only the 14 show-agnostic structural features (`hand` variant) so the model transfers to brand-new shows.
   - *Validation:* 5-fold leave-episode-out cross-validation achieves **0.789 AUC** on hooks and **0.826 AUC** on cliffhangers (`models/cv_report.json`).

2. **Cold-Viewer Simulator & Curiosity-Gap Score (`judge.py`)**
   - *Insight:* "Makes sense on its own to a stranger" is the hardest criterion for keyword rules. We show an LLM *only* the candidate clip transcript and ask it to list **plot-breaking past confusions** (unexplained names/events that block comprehension) and **urgent future questions** (open stakes at the cut), plus 0–10 ratings for standalone clarity and cliffhanger pull.

3. **Position-Debiased Pairwise Tournament (`rank.py`)**
   - *Insight:* Absolute 0–10 LLM scores are poorly calibrated across episodes. For the top 12 finalists, we run a round-robin tournament (`66` pairs × `2` orders = `132` calls). A clip only wins a match if the LLM picks it when presented as `(A, B)` **and** when presented as `(B, A)`; disagreements count as a `0.5` draw.

---

## Quality Checking & Where It Fails

Run the checks yourself:
```bash
python eval.py sanity --input "<folder>"     # 100/100 promos pass hard constraints (20-60s, no split dialogue, valid bounds)
python eval.py score                         # AUC & Precision@6 across 24 blind-shuffled clips in gold/gold.csv
```

### Real Failure Modes (Honest Assessment)
- **Visual-only hooks without charged verbs:** In Episode 50 (`00:49.80 -> 01:18.50`, ranked #5), the ending is an A+ scandal reveal (Agnes catches Catherine in bed with the shirtless priest), but the opening 3 seconds (`"I've told myself that since the dawn in your bedchamber..."`) is a quiet conversational hook (`rule hook = 2.0`). It survives only because the cliffhanger and curiosity gap are huge.
- **Emotionally intense resolutions mistaken for cliffhangers:** Episode 16 (`00:41.20 -> 01:23.19`, ranked #7) opens with Alexander choking a thug against a wall (great hook), but ends on a quiet romantic gaze accompanied by `[SOUND] Slow, resonant emotional cello melody swelling to a dramatic cliffhanger hold`. Because the script writer tagged the sound cue with the word `"cliffhanger"`, both the rules and the learned model score the ending high even though the fight already resolved at `00:56`.
- **Filler episodes produce weak second picks:** Because `--per-episode 2` forces two non-overlapping picks per episode, quiet transition episodes (e.g., Ep 8 `#100`, Ep 48 `#99`) yield low-scoring clips (`score <= 3.1`) that an editor should skip. In production, filter by `score >= 5.5` instead of taking a fixed quota per episode.
- **Single-segment only:** Multi-segment mashups (stitching a 3-second slap from `01:20` onto the start of a buildup at `00:10`) are not implemented; every output promo is one continuous window.
