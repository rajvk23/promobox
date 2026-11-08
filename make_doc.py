"""Generate the 2-page Solution Doc PDF (solution_doc.pdf) using fpdf."""
from fpdf import FPDF


def clean(t):
    return (t.replace("—", "-").replace("–", "-").replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"').replace("→", "->").replace("×", "x")
             .replace("≥", ">=").replace("≤", "<=").replace("±", "+/-"))


class Doc(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5, clean(f"Kuku AI Generalist Assignment - Problem 1: Find the Promos   |   Page {self.page_no()} of 2"), 0, 0, "C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(20, 45, 95)
        self.cell(0, 5.5, clean(text), ln=1)
        self.set_draw_color(200, 210, 225)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(1.2)

    def p(self, text, bold_prefix=""):
        self.set_font("Helvetica", "", 8.4)
        self.set_text_color(25, 25, 25)
        if bold_prefix:
            self.set_font("Helvetica", "B", 8.4)
            self.write(4.0, clean(bold_prefix + " "))
            self.set_font("Helvetica", "", 8.4)
        self.multi_cell(0, 4.0, clean(text))
        self.ln(0.8)

    def bullet(self, bold_prefix, text):
        self.set_x(self.l_margin + 2)
        self.set_font("Helvetica", "B", 8.3)
        self.set_text_color(25, 25, 25)
        self.write(3.9, clean("- " + bold_prefix + " "))
        self.set_font("Helvetica", "", 8.3)
        self.multi_cell(self.w - self.r_margin - self.l_margin - 2, 3.9, clean(text))
        self.ln(0.5)


pdf = Doc(orientation="P", unit="mm", format="A4")
pdf.set_margins(12, 10, 12)
pdf.set_auto_page_break(auto=True, margin=11)

# ============================== PAGE 1 ==============================
pdf.add_page()
pdf.set_font("Helvetica", "B", 13.5)
pdf.set_text_color(15, 30, 70)
pdf.cell(0, 6.5, clean("Kuku AI Generalist Assignment - Problem 1: Find the Promos"), ln=1)
pdf.set_font("Helvetica", "", 8.5)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 4.5, clean("Prototype Repo: https://github.com/rajvk23/promobox   |   Raw Outputs: https://github.com/rajvk23/promobox/tree/main/outputs"), ln=1)
pdf.ln(1.5)

pdf.h1("1. Problem Choice, Framing & Explicit Assumptions")
pdf.p("Why Problem 1: Reviewing 50 episodes by hand to find 20-60s ad hooks is a high-volume editorial bottleneck that repeats every week across languages. Because Kuku's screenplays already carry precise timestamps, speaker tags, emotions, and camera/sound cues, we can solve candidate discovery in deterministic code and reserve AI strictly for semantic judgment.")
pdf.bullet("Single continuous windows (20-60s):", "While promos can stitch multiple segments, continuous scenes preserve dramatic pacing without jarring jump-cuts. We scope v1 to continuous windows aligned strictly to dialogue/action boundaries (never cutting a spoken line in half), with 25-50s treated as the short-form sweet spot.")
pdf.bullet("Up to 2 non-overlapping candidates per episode:", "Gives editors 100 total candidates across the 50-episode pack, ranked globally so the acquisition team knows exactly which 5-10 clips to test first.")
pdf.bullet("Zero-dependency fallback:", "If an API key is missing or rate-limited during a live run, every stage degrades gracefully to an offline model/heuristic so the tool never crashes on fresh inputs.")

pdf.h1("2. Architecture: Code vs. Self-Supervised ML vs. LLM vs. Human")
pdf.p("Rather than dumping entire screenplays into an LLM (which hallucinates timestamps and scales poorly), we use a 4-stage funnel where cheap stages filter 16,600+ windows down to 12 finalists:")
pdf.bullet("Stage 1 - Deterministic Parser & Window Proposer (Code, promo_finder.py):", "Parses character dialogue, parenthetical emotions, ACTION, CAMERA, and SOUND FX across all 50 episodes (~16,600 valid 20-60s windows). Guarantees exact timestamps and zero split lines.")
pdf.bullet("Stage 2a - Structural & Lexicon Rules (Code):", "Scores every window on Hook (first 3s: slaps, accusations, exclamations, intense emotion tags, penalising weak back-channel openers like 'Yes/And'), Cliffhanger (last 6s: episode-end stings, unanswered questions, threats, penalising resolved lines), Standalone clarity (scene-start bonus, cast-size cap, early-episode bonus), and Intensity.")
pdf.bullet("Stage 2b - Novelty 1: Self-Supervised Hook/Cliffhanger Classifier (ML, learn.py):", "Every episode opening (first 3s) is a writer-chosen hook and every episode ending (last 6s) is a writer-chosen cliffhanger. This gives 50 free positive examples per class with zero manual labelling. We train a scikit-learn Logistic Regression classifier on 14 show-agnostic structural features (blended 50/50 with rules). In 5-fold leave-episode-out cross-validation, it achieves 0.789 AUC on hooks and 0.826 AUC on cliffhangers.")
pdf.bullet("Stage 3 - Novelty 2: Cold-Viewer Simulator & Curiosity-Gap Score (LLM, judge.py):", "Rules are worst at judging 'makes sense on its own.' For only the top 4 windows per episode (200 total), an LLM (gemini-3.1-flash-lite, swappable via LLM_MODEL / LLM_BASE_URL) reads the clip as a stranger who has never seen the show, listing plot-breaking past_confusions vs. urgent future_questions and rating standalone clarity & cliffhanger pull.")
pdf.bullet("Stage 4 - Novelty 3: Position-Debiased Pairwise Tournament (LLM, rank.py):", "Absolute 0-10 LLM scores are uncalibrated across episodes. The top 12 finalists enter a round-robin tournament (66 pairs x 2 orders = 132 calls). Clip A beats Clip B only if the LLM picks A when shown (A, B) AND when shown (B, A); order-sensitive flips become 0.5 draws. Human editors make the final cut.")

pdf.h1("3. Scale & Unit Economics at Production Volume")
pdf.p("On all 50 episodes, Stages 1-2 evaluate ~16,600 windows in 2.5 seconds on CPU ($0). Stage 3 makes 200 short LLM calls and Stage 4 makes 132 calls (~332 calls/show, ~140k input + 25k output tokens). Using Gemini Flash-Lite or Flash free/paid tiers, a full 50-episode show costs ~$0.02-$0.05 and runs in ~40 seconds with thread parallelism (and <2s on cached re-runs). At 20 new shows/week, total AI cost is under $1/week.")

# ============================== PAGE 2 ==============================
pdf.add_page()
pdf.h1("4. Top Ranked Promos: Which to Post First & Why")
pdf.p("Across all 50 episodes, our pairwise tournament and composite scorer surface three distinct promo archetypes to post first on Reels/TikTok/Shorts (full 100-clip table + transcripts in outputs/promos.md):")
pdf.bullet("#1 (Post First - Visceral Dungeon Torture): Ep 31, 02:02.00 -> 02:40.00 (38.0s, Score 6.12, 11.0/11 Wins).", "Hook: '[ACTION] Catherine's smug expression freezes... as the blow lands squarely' followed by Juliana's defiance ('If you were truly powerful, Catherine, you wouldn't need me to bow'). Cliffhanger: Catherine torches Juliana's late mother's book and thrusts a searing branding iron inches from Juliana's eyes. Why first: Instant physical conflict in 2s, zero backstory needed to understand captor vs. prisoner, and an unbearable visual cut.")
pdf.bullet("#2 (Post Second - Class Humiliation & Physical Strike): Ep 24, 00:33.80 -> 01:10.50 (36.7s, Score 6.44, 8.0 Wins).", "Hook: '[ACTION] The noblewoman suddenly rears back her riding crop and strikes Juliana violently across the torso.' Cliffhanger: As Juliana drops to her knees before a soot-filled fireplace, the noblewoman thrusts the crop into young maid Prudence's face: 'Do you want the same punishment?' Why second: Classic Cinderella-trope abuse that hooks in 1.3s and ends on an immediate threat to an innocent bystander.")
pdf.bullet("#3 (Post Third - Altar Betrayal, Series Premise): Ep 1, 01:47.50 -> 02:17.50 (30.0s, Score 6.27, 8.0 Wins).", "Hook: Clarissa gloating in gold at the altar: 'You are the daughter people tolerate because they have to. I am the daughter they choose.' Cliffhanger: Groom rejects Juliana as 'low value' while Juliana clenches her mother's bridal gown in cold fury as the string drone peaks. Why third: Establishes the core series premise for cold audiences with zero spoilers of later twists.")
pdf.bullet("#4 & #5 (High-Stakes Conspiracy & Scandal):", "Ep 30 (00:08.80 -> 00:39.80, Score 6.00, 7.5 Wins: 'You have a rat in your cellar, Rupert... you burn them!') and Ep 50 (00:49.80 -> 01:18.50, Score 6.00, 6.0 Wins: maid Agnes catches Lady Catherine in bed with the shirtless priest Father Thomas, gasping in terror before cut to black).")

pdf.h1("5. Quality Checking & Where the Tool Picks Badly")
pdf.p("Automated Sanity & Human-Labelled Benchmark (eval.py):", "All 100 output promos pass hard boundary checks (0/100 outside 20-60s, 0/100 cutting a spoken line in half). On a stratified 24-clip evaluation set (gold/gold.csv, 11 good / 13 bad), all four variants achieve Precision@6 = 1.00, with ROC-AUC improving from 0.972 (rules only) to 0.979 (rules + learned ML / full LLM) and 0.983 (full offline).")
pdf.p("Honest Failure Modes (Where Our Tool Picks Badly):", "Inspecting the full 100-clip ranking reveals four concrete weaknesses:")
pdf.bullet("1. Stage-direction keyword leakage on resolved scenes (Ep 16, 00:41.20 -> 01:23.19, Ranked #7):", "Opens with a great hook (Alexander choking a thug: 'Who sent you?'), but the thug confesses and flees by 00:56. The last 25 seconds are a quiet, tender gaze between Alexander and Juliana. However, because the screenplay's final SOUND FX line literally contains the words 'swelling to a dramatic cliffhanger hold', the rule scorer gives cliff=6.5 and the clip sneaks into the top 10 despite ending on an emotional resolution.")
pdf.bullet("2. Quiet conversational openers rescued only by huge endings (Ep 50, 00:49.80 -> 01:18.50, Ranked #5):", "Ends on arguably the best scandal cliffhanger in the pack (Agnes witnessing Catherine and the priest), but opens on 'I've told myself that since the dawn in your bedchamber...' (rule hook = 2.0). A real TikTok editor would trim the first 10s or prepend a 2s flash-forward of Agnes's gasp.")
pdf.bullet("3. Forced quota on low-conflict transition episodes (Ranks #85-#100):", "Because --per-episode 2 forces two picks per episode, quiet world-building episodes (e.g. Ep 8 #100 at 00:50.10 -> 01:18.00, Score 2.56, fireplace herbal chat; Ep 48 #99, Score 3.07) produce flat clips. In production, editors should filter by Score >= 5.5 rather than taking a fixed per-episode quota.")
pdf.bullet("4. Late-series lore density (Ep 45 #12, Ep 42 #9):", "Even when dramatic, late-episode reveals ('You poisoned my mother. You tortured my wife.') spoil major twists and rely on 40 episodes of buildup.")

pdf.h1("6. AI Tools Used & Where AI Got It Wrong During Development")
pdf.p("Tools used: Claude / Gemini in Antigravity IDE for architecture design, scikit-learn model iteration, and Gemini 3.1 Flash-Lite via REST API for Stage 3-4 runtime scoring.")
pdf.bullet("Mistake 1 - Confounded Self-Supervised Labels:", "In our first version of Novelty 1, TF-IDF + Logistic Regression achieved a suspiciously high 0.85+ AUC, and its top features were 'ACT' (action block) and plot words like 'ten days'. Inspecting the data showed two bugs: (a) episode edges almost always start/end on an ACTION block, so random mid-episode dialogue negatives taught the model 'ACTION = hook'; and (b) roughly 40% of episode endings in this romance show are tender fireplace moments (Ep 6, 7, 8), not cliffhangers! Fix: We matched negative samples by event kind, filtered positive training edges by the rule median, and shipped only the 14 show-agnostic structural features.")
pdf.bullet("Mistake 2 - LLM List-Padding in Curiosity-Gap Prompting:", "In our first test of Novelty 2 on Episode 1, Gemini returned exactly 4 past_confusions and 4 future_questions on every single clip (asking lore trivia like 'Why is she wearing her mother's gown?'), which pegged our count formula to 8.0/10 across the board. Fix: We tightened the prompt to restrict past_confusions strictly to plot-breaking missing context (returning [] when self-explanatory) and blended count ratios with explicit 0-10 clarity/pull ratings.")

out_path = "solution_doc.pdf"
pdf.output(out_path)
print("Wrote", out_path, "pages:", pdf.page_no())
