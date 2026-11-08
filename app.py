"""Streamlit app: Modern Studio Dashboard for Microdrama Promo Finder.
Run with: python -m streamlit run app.py
"""
import html
import json
import re
import streamlit as st

import promo_finder as pf
import run as R
import llm

st.set_page_config(
    page_title="KUKU • Microdrama Promo Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Dark Cinema Theme, custom cards, clean typography, badge badges)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    .main {
        background: radial-gradient(circle at top right, #181926 0%, #0d0e15 100%);
    }

    /* Hero Header */
    .hero-container {
        padding: 1.6rem 2rem;
        background: linear-gradient(135deg, rgba(235, 34, 76, 0.12) 0%, rgba(138, 43, 226, 0.08) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 2rem;
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #F43F5E 50%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Metric Badges */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #E2E8F0;
    }

    /* Promo Card */
    .promo-card {
        background: rgba(20, 22, 34, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    .promo-card:hover {
        border-color: rgba(244, 63, 94, 0.4);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(244, 63, 94, 0.15);
    }

    .badge-rank {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.85rem;
        background: linear-gradient(135deg, #F43F5E, #E11D48);
        color: white;
        letter-spacing: 0.5px;
    }
    .badge-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-score {
        font-size: 0.82rem;
        color: #34D399;
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.25);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-wins {
        font-size: 0.82rem;
        color: #FBBF24;
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.25);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
    }

    .hook-box {
        background: rgba(244, 63, 94, 0.05);
        border-left: 3px solid #F43F5E;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #F8FAFC;
    }
    .cliff-box {
        background: rgba(168, 85, 247, 0.05);
        border-left: 3px solid #A855F7;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #F8FAFC;
    }
    .why-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.86rem;
        color: #CBD5E1;
        margin-top: 10px;
    }

    /* Consistent Transcript formatting - No weird syntax highlighting */
    .transcript-container {
        background: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.65;
        max-height: 380px;
        overflow-y: auto;
        margin-top: 12px;
        color: #E2E8F0;
    }
    .transcript-line {
        margin-bottom: 6px;
        word-break: break-word;
    }
    .t-time {
        color: #64748B;
        font-weight: 500;
        user-select: none;
    }
    .t-speaker {
        color: #38BDF8;
        font-weight: 700;
    }
    .t-emotion {
        color: #94A3B8;
        font-style: italic;
    }
    .t-dialogue {
        color: #F1F5F9;
    }
    .t-action {
        color: #FBBF24;
    }
    .t-sound {
        color: #F43F5E;
        font-weight: 600;
    }
    .t-camera {
        color: #A855F7;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 class="hero-title">🎬 KUKU Promo Finder Studio</h1>
            <p class="hero-sub">AI-Powered Short-Form Ad Extractor • Boundary-Safe • Curiosity-Gap Evaluated</p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="metric-pill">🎯 20–60s Windows</span>
            <span class="metric-pill">⚡ Rules + ML + LLM</span>
            <span class="metric-pill">🏆 Pairwise Tournament</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    env_key = llm.cfg()["key"]
    if not env_key:
        user_key = st.text_input("🔑 Gemini / OpenAI API Key (optional)", type="password",
                                 help="Paste a Gemini API key to enable live Cold-Viewer & Pairwise LLM judging. Leave blank for instant offline ML mode.")
        if user_key:
            import os
            os.environ["LLM_API_KEY"] = user_key.strip()

    use_llm = st.toggle("Enable LLM Evaluation", value=llm.available(),
                        help="Runs Novelty 2 (Curiosity-Gap Cold-Viewer) and Novelty 3 (Pairwise Tournament Judge)")
    if use_llm and not llm.available():
        st.warning("⚠️ No API key set — running fast offline fallbacks (Rules + Learned ML + Condorcet Tournament).")

    use_clf = st.toggle("Use Learned Edge Model", value=True,
                        help="Novelty 1: Self-supervised classifier trained on show openings/endings")
    
    st.markdown("---")
    st.markdown("### 🎚️ Candidate Filtering")
    per_ep = st.slider("Max Promos per Episode", min_value=1, max_value=4, value=2)
    pool = st.slider("Finalists in Tournament", min_value=2, max_value=24, value=12)

    st.markdown("---")
    st.markdown("### 💡 Quick Instructions")
    st.caption("1. Upload any episode `.txt` screenplay.")
    st.caption("2. Click **Extract Promos**.")
    st.caption("3. Review the ranked hooks, cliffhangers, and formatted transcripts.")

# Main Interface: File Upload
uploaded_files = st.file_uploader(
    "Drop Episode Screenplay Files Here (.txt)",
    type=["txt"],
    accept_multiple_files=True,
    help="Upload screenplay files in Kuku's timestamped format"
)

# Render Transcript Helper with deliberate, correct semantic colors
def render_formatted_transcript(transcript_text):
    lines = transcript_text.strip().split("\n")
    html_lines = []
    
    for line in lines:
        if not line.strip():
            continue
        
        # Extract timestamp [mm:ss.ss]
        time_m = re.match(r"^\[(\d+:\d+\.\d+)\]\s*(.*)$", line)
        if time_m:
            ts_str, rest = time_m.group(1), time_m.group(2)
            ts_html = f'<span class="t-time">[{ts_str}]</span> '
        else:
            ts_html = ""
            rest = line

        # Parse type of event
        if rest.startswith("[ACTION]"):
            content = html.escape(rest[8:].strip())
            html_lines.append(f'<div class="transcript-line">{ts_html}<span class="t-action">🎬 ACTION:</span> {content}</div>')
        elif rest.startswith("[SOUND]"):
            content = html.escape(rest[7:].strip())
            html_lines.append(f'<div class="transcript-line">{ts_html}<span class="t-sound">🔊 SOUND:</span> {content}</div>')
        elif rest.startswith("[CAMERA]"):
            content = html.escape(rest[8:].strip())
            html_lines.append(f'<div class="transcript-line">{ts_html}<span class="t-camera">🎥 CAMERA:</span> {content}</div>')
        else:
            # Dialogue: SPEAKER (Emotion): "Text"
            dlg_m = re.match(r"^([A-Z0-9\s\.\'\-]+?)(?:\s*\((.*?)\))?:\s*(.*)$", rest)
            if dlg_m:
                spk, emo, text = dlg_m.group(1).strip(), dlg_m.group(2), dlg_m.group(3)
                spk_html = f'<span class="t-speaker">{html.escape(spk)}</span>'
                emo_html = f' <span class="t-emotion">({html.escape(emo)})</span>' if emo else ''
                txt_html = f'<span class="t-dialogue">: {html.escape(text)}</span>'
                html_lines.append(f'<div class="transcript-line">{ts_html}{spk_html}{emo_html}{txt_html}</div>')
            else:
                html_lines.append(f'<div class="transcript-line">{ts_html}{html.escape(rest)}</div>')

    return f'<div class="transcript-container">{"".join(html_lines)}</div>'


if uploaded_files:
    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        start_btn = st.button("🚀 Extract Promos", type="primary", use_container_width=True)

    if start_btn:
        eps = []
        errors = []
        for f in uploaded_files:
            try:
                content = f.read().decode("utf-8", errors="replace")
                eps.append(pf.parse_text(content, f.name))
            except Exception as e:
                errors.append(f"{f.name}: {str(e)}")

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")

        if eps:
            with st.spinner("Analyzing dramatic tension, proposing cut boundaries, and evaluating tournament..."):
                ranked, meta, info = R.find_promos(eps, use_llm, use_clf, per_ep, pool)

            st.success(f"✨ Found and ranked **{len(ranked)}** ad promo clips across **{len(eps)}** episode(s)!")

            # Quick summary metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Total Candidates Evaluated", sum(i[3] for i in info))
            with m_col2:
                st.metric("Model Engine", "Rules + ML + LLM" if meta.get("llm") else "Rules + Learned ML")
            with m_col3:
                st.metric("Tournament Format", meta.get("tournament", "offline"))
            with m_col4:
                top_dur = f"{ranked[0].end - ranked[0].start:.1f}s" if ranked else "N/A"
                st.metric("Top Pick Duration", top_dur)

            st.markdown("### 🏆 Ranked Promo Recommendations")

            for rank_num, c in enumerate(ranked, 1):
                dur = round(c.end - c.start, 1)
                
                # Card Container
                st.markdown(f"""
                <div class="promo-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span class="badge-rank">RANK #{rank_num}</span>
                            <span style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">Episode {c.episode}</span>
                            <span class="badge-time">⏱️ {pf.fmt(c.start)} → {pf.fmt(c.end)} ({dur}s)</span>
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <span class="badge-score">Score: {c.final:.2f}</span>
                            <span class="badge-wins">Wins: {c.wins}</span>
                        </div>
                    </div>
                    
                    <div class="hook-box">
                        <strong>🪝 HOOK (First 3s):</strong><br/>
                        {html.escape(c.hook_line)}
                    </div>
                    
                    <div class="cliff-box">
                        <strong>⚡ CLIFFHANGER (Cut Point):</strong><br/>
                        {html.escape(c.cliff_line)}
                    </div>
                    
                    <div class="why-box">
                        <strong>💡 Why It Works:</strong> {html.escape(R.why_line(c))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Expandable details for Clean Transcript & Diagnostics
                with st.expander(f"📜 View Full Timestamped Transcript & Diagnostics (#{rank_num})", expanded=False):
                    diag_col1, diag_col2 = st.columns([1, 1])
                    with diag_col1:
                        st.markdown("**Sub-scores Breakdown:**")
                        st.json({
                            "Composite Final": c.final,
                            "Hook Strength": c.scores.get("hook"),
                            "Cliffhanger Pull": c.scores.get("cliff"),
                            "Standalone Clarity": c.scores.get("standalone"),
                            "Dramatic Intensity": c.scores.get("intensity")
                        })
                    with diag_col2:
                        if c.gap:
                            st.markdown(f"**Curiosity Gap ({c.gap.get('source', 'engine')}):**")
                            st.write(f"- **Past Confusions (0 is best):** {c.gap.get('past_confusions', [])}")
                            st.write(f"- **Future Questions (Higher is better):** {c.gap.get('future_questions', [])}")
                    
                    st.markdown("**Formatted Screenplay Transcript:**")
                    st.markdown(render_formatted_transcript(c.transcript), unsafe_allow_html=True)

            # Export download
            st.markdown("---")
            export_rows = [{
                "rank": i,
                "episode": c.episode,
                "start": pf.fmt(c.start),
                "end": pf.fmt(c.end),
                "duration_s": round(c.end - c.start, 1),
                "hook": c.hook_line,
                "cliffhanger": c.cliff_line,
                "why_it_works": R.why_line(c),
                "score": c.final,
                "tournament_wins": c.wins
            } for i, c in enumerate(ranked, 1)]
            
            st.download_button(
                "📥 Download Ranked Promos (JSON)",
                data=json.dumps(export_rows, indent=2),
                file_name="kuku_ranked_promos.json",
                mime="application/json"
            )
else:
    st.info("👆 Please upload one or more episode screenplay `.txt` files above to extract ad promos.")
