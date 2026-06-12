import streamlit as st
import mlx_whisper
import os
import json
import re
import plotly.express as px
import pandas as pd

from agents.graph import closingforce_graph
from agents.speaker_labeler import speaker_labeler_agent
from agents.deal_intel import deal_intel_agent
from agents.objection_mapper import objection_mapper_agent
from agents.proposal_crafter import proposal_crafter_agent

# =============================================================================
# PREMIUM DARK THEME + STYLING
# =============================================================================
st.set_page_config(
    page_title="Deal War Room",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium SaaS dark theme
st.markdown("""
<style>
    /* === Global Dark Theme === */
    .stApp {
        background-color: #0F172A;
        color: #F1F5F9;
    }
    
    /* Hide default Streamlit header/footer for cleaner look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* === Hero Header === */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E2937 100%);
        padding: 2.25rem 2.75rem;
        border-radius: 20px;
        margin-bottom: 1.75rem;
        border: 1px solid #334155;
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1.05;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        font-weight: 400;
    }
    
    .hero-badge {
        background: linear-gradient(90deg, #14B8A6, #0EA47A);
        color: #0F172A;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 0.6rem;
        letter-spacing: 0.3px;
    }
    
    /* === Rocket Animation for Loading === */
    .rocket-loader {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        font-size: 1.1rem;
        color: #14B8A6;
        font-weight: 600;
    }
    
    .rocket {
        display: inline-block;
        animation: fly 1.2s ease-in-out infinite;
        font-size: 1.4rem;
    }
    
    @keyframes fly {
        0%, 100% { transform: translateY(0) rotate(-10deg); }
        50% { transform: translateY(-12px) rotate(10deg); }
    }
    
    /* === Responsive Design === */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.75rem; }
        .hero-container { padding: 1.5rem 1.25rem; }
        .metric-card { padding: 0.85rem !important; }
    }
    
    @media (max-width: 480px) {
        .hero-title { font-size: 1.55rem; }
    }
    
    /* === Cards === */
    .premium-card {
        background-color: #1E2937;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    .premium-card-title {
        color: #14B8A6;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* === Buttons === */
    .stButton > button {
        background-color: #14B8A6;
        color: #0F172A;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #0EA47A;
        transform: translateY(-1px);
    }
    
    /* === Download Buttons (clean, visible dark-theme styling) === */
    .stDownloadButton > button {
        background-color: #14B8A6;
        color: #0F172A;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton > button:hover {
        background-color: #0EA47A;
        transform: translateY(-1px);
    }
    
    /* Secondary button style */
    .secondary-button > button {
        background-color: transparent !important;
        border: 1px solid #475569 !important;
        color: #E2E8F0 !important;
    }
    
    /* === Metrics === */
    .metric-card {
        background-color: #1E2937;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #14B8A6;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    
    /* === Progress / Status === */
    .agent-step {
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.35rem;
        font-size: 0.9rem;
    }
    
    .agent-active {
        background-color: #14B8A620;
        border: 1px solid #14B8A6;
        color: #14B8A6;
        font-weight: 600;
    }
    
    .agent-done {
        background-color: #1E2937;
        border: 1px solid #334155;
        color: #94A3B8;
    }
    
    /* === JSON & Text Improvements === */
    .stJson, .stTextArea textarea {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;   /* Force white text in text areas */
    }
    
    /* Force visible text in text_area */
    .stTextArea textarea,
    .stTextArea textarea:focus {
        color: #F8FAFC !important;
        caret-color: #F8FAFC !important;
    }
    
    /* Better spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* === Make File Uploader Visible on Dark Theme === */
    .stFileUploader label {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
    }
    
    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        background-color: #1E2937 !important;
        border-color: #475569 !important;
    }
    
    .stFileUploader div[data-testid="stFileUploadDropzone"] p {
        color: #CBD5E1 !important;
    }

    /* === Dark blue upload/browse button (fix white-on-white) === */
    .stFileUploader div[data-testid="stFileUploadDropzone"] button,
    .stFileUploader button,
    .stFileUploader [data-testid="stBaseButton"] button {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* === Make Expander Headers Visible + Persistent Dark Blue (no white change on open) === */
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] details[open] summary {
        background-color: #1E3A8A !important;
        color: #F1F5F9 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
    }
    
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] details summary p,
    div[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] details summary * {
        color: #F1F5F9 !important;
    }
    
    /* Extra safety for expander header text */
    .stExpander details summary {
        background-color: #1E3A8A !important;
        color: #F1F5F9 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HERO HEADER (Premium SaaS)
# =============================================================================
st.markdown("""
<div class="hero-container">
    <div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 class="hero-title">Deal War Room</h1>
            <p class="hero-subtitle">Enterprise-grade AI intelligence for high-stakes sales conversations</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CONTROLS
# =============================================================================
col1, col2 = st.columns([3, 1])

with col1:
    audio_file = st.file_uploader(
        "Upload sales call recording (.wav)", 
        type=["wav"],
        label_visibility="visible"
    )

with col2:
    st.write("")  # spacing
    run_clicked = st.button("▶️ Run Full Workflow", use_container_width=True, type="primary")
    demo_clicked = st.button("🎬 Load Demo Call", use_container_width=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.raw_transcript = None
    st.session_state.is_demo = False

# =============================================================================
# PROCESSING LOGIC WITH LIVE PROGRESS
# =============================================================================
def process_call_with_progress(audio_path: str, is_demo: bool = False):
    """Process the call with a beautiful animated loading experience + per-agent progress."""
    
    # Create a placeholder for the animated loader
    loader_placeholder = st.empty()
    
    # Show beautiful animated rocket loader
    loader_placeholder.markdown("""
    <div class="rocket-loader" style="margin: 1.5rem 0; padding: 1rem; background: #1E2937; border-radius: 12px; border: 1px solid #334155;">
        <span class="rocket">🚀</span> 
        <span>Analyzing call with specialized AI agents...</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Transcription
    with st.status("Transcribing audio with mlx-whisper...", expanded=True) as status:
        result = mlx_whisper.transcribe(audio_path)
        raw_transcript = result["text"]
        status.update(label="✅ Transcription complete", state="complete")
    
    st.session_state.raw_transcript = raw_transcript
    
    # Step 2+: Run agents one by one with visible progress
    with st.status("Running 4 specialized agents...", expanded=True) as status:
        
        # Speaker Labeling
        st.write("🔖 Running **SpeakerLabelerAgent**...")
        structured_transcript = speaker_labeler_agent(raw_transcript)
        
        # Deal Intelligence
        st.write("📊 Running **DealIntelAgent**...")
        deal_intel = deal_intel_agent(structured_transcript)
        
        # Objection Mapping
        st.write("⚠️ Running **ObjectionMapperAgent**...")
        objections = objection_mapper_agent(structured_transcript)
        
        # Proposal Creation
        st.write("✉️ Running **ProposalCrafterAgent**...")
        proposal = proposal_crafter_agent(structured_transcript, deal_intel, objections)
        
        # Final Summary
        final_summary = f"""Deal Intelligence:\n{deal_intel}\n\nObjections:\n{objections}\n\nProposal:\n{proposal}"""
        
        status.update(label="✅ All agents completed", state="complete")
    
    # Save everything to session state
    st.session_state.results = {
        "structured_transcript": structured_transcript,
        "deal_intel": deal_intel,
        "objections": objections,
        "proposal": proposal,
        "final_summary": final_summary
    }
    st.session_state.is_demo = is_demo
    
    # Clear the animated rocket loader once processing is complete
    loader_placeholder.empty()

# Handle button clicks
if run_clicked and audio_file:
    os.makedirs("data", exist_ok=True)
    temp_path = "data/temp_call.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    process_call_with_progress(temp_path, is_demo=False)

elif demo_clicked:
    demo_path = "data/test_call.wav"
    if os.path.exists(demo_path):
        process_call_with_progress(demo_path, is_demo=True)
    else:
        st.error("Demo file not found. Please ensure `data/test_call.wav` exists.")

# =============================================================================
# HOW IT WORKS (Premium Section)
# =============================================================================
with st.expander("How Deal War Room Works", expanded=False):
    st.markdown("#### Four specialized AI agents work together in sequence")
    
    cols = st.columns(4)
    
    agents = [
        ("1", "Speaker Labeler", "Accurately identifies Sales Rep vs Customer in the conversation"),
        ("2", "Deal Intel", "Extracts MEDDPICC framework, pain points, and buying signals"),
        ("3", "Objection Mapper", "Surfaces every customer concern and hesitation"),
        ("4", "Proposal Crafter", "Generates a personalized, professional follow-up proposal")
    ]
    
    for i, (num, title, desc) in enumerate(agents):
        with cols[i]:
            st.markdown(f"""
            <div style="background:#1E2937; border:1px solid #334155; border-radius:12px; padding:1rem; height:100%;">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">{num}</div>
                <div style="font-weight:600; color:#14B8A6; margin-bottom:0.35rem;">{title}</div>
                <div style="font-size:0.8rem; color:#94A3B8; line-height:1.35;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# RESULTS DISPLAY + ANALYTICS
# =============================================================================
if st.session_state.results:
    state = st.session_state.results
    raw_transcript = st.session_state.raw_transcript or ""
    
    st.markdown("---")
    
    # === COMPUTE ANALYTICS ===
    # Parse objections count
    try:
        objections_data = json.loads(state.get("objections", "{}"))
        objections_list = objections_data.get("objections", []) if isinstance(objections_data, dict) else []
        num_objections = len(objections_list)
    except:
        num_objections = 0
    
    # Parse Deal Intel
    try:
        deal_data = json.loads(state.get("deal_intel", "{}"))
        if not isinstance(deal_data, dict):
            deal_data = {}
    except:
        deal_data = {}
    
    filled_fields = sum(1 for v in deal_data.values() if v and str(v).strip() and str(v).strip() != '""')
    pain_points = deal_data.get("identified_pain", "") or ""
    
    # Speaker talk time distribution
    speaker_times = {}
    lines = raw_transcript.split("\n")
    for line in lines:
        if ":" in line:
            speaker = line.split(":")[0].strip()
            text = ":".join(line.split(":")[1:]).strip()
            word_count = len(text.split())
            speaker_times[speaker] = speaker_times.get(speaker, 0) + word_count
    
    # Call Quality Score (heuristic)
    base_score = 65
    base_score += min(filled_fields * 4, 20)           # + up to 20 for deal intel
    base_score -= min(num_objections * 3, 25)          # - for objections
    base_score += 10 if pain_points and len(pain_points) > 20 else 0
    call_quality_score = max(35, min(98, base_score))
    
    # === CALL QUALITY SCORE + INSIGHTS ===
    st.markdown("### 📊 Call Quality Score & Insights")
    
    # Big Score
    score_col, insights_col = st.columns([1, 2])
    
    with score_col:
        st.markdown(f"""
        <div style="background:#1E2937; border:1px solid #334155; border-radius:16px; padding:1.5rem; text-align:center;">
            <div style="font-size:0.85rem; color:#94A3B8; margin-bottom:0.5rem;">CALL QUALITY SCORE</div>
            <div style="font-size:3.2rem; font-weight:800; color:#14B8A6; line-height:1;">{call_quality_score}</div>
            <div style="font-size:0.8rem; color:#64748B;">/ 100</div>
            <div style="margin-top:0.75rem;">
                <progress value="{call_quality_score}" max="100" style="width:100%; height:8px; accent-color:#14B8A6;"></progress>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col:
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card" style="height:110px;">
                <div style="font-size:2rem; font-weight:700; color:#14B8A6;">{num_objections}</div>
                <div style="font-size:0.8rem; color:#94A3B8;">Objections Found</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
            <div class="metric-card" style="height:110px;">
                <div style="font-size:2rem; font-weight:700; color:#14B8A6;">{filled_fields}</div>
                <div style="font-size:0.8rem; color:#94A3B8;">Deal Signals Captured</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            next_action = "Send proposal & schedule follow-up" if num_objections < 5 else "Address objections first"
            st.markdown(f"""
            <div class="metric-card" style="height:110px; background:#1E2937;">
                <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:4px;">NEXT BEST ACTION</div>
                <div style="font-size:0.9rem; line-height:1.3; color:#E2E8F0;">{next_action}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === SPEAKER TALK TIME CHART ===
    if speaker_times:
        st.markdown("### 🗣️ Speaker Talk Time Distribution")
        
        df_speakers = pd.DataFrame({
            "Speaker": list(speaker_times.keys()),
            "Words Spoken": list(speaker_times.values())
        })
        
        fig = px.pie(
            df_speakers, 
            values="Words Spoken", 
            names="Speaker",
            color_discrete_sequence=["#14B8A6", "#0EA47A", "#64748B"],
            hole=0.45
        )
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    # === KEY MOMENTS HIGHLIGHTED ===
    if raw_transcript:
        important_keywords = ["pain", "problem", "issue", "frustrat", "expensive", "slow", "nobody", "follow up", "cost", "budget"]
        highlighted_lines = []
        
        for line in raw_transcript.split("\n"):
            line_lower = line.lower()
            if any(kw in line_lower for kw in important_keywords):
                highlighted_lines.append(line)
        
        if highlighted_lines:
            st.markdown("### 🔥 Key Moments (Potential Pain Points)")
            for line in highlighted_lines[:5]:  # Limit to top 5
                st.markdown(f"• {line}")
    
    # === STRUCTURED TRANSCRIPT (Speaker Labeled) ===
    with st.expander("🔖 Structured Transcript (Speaker Labeled)", expanded=True):
        transcript_text = state.get("structured_transcript", "") or ""
        if transcript_text:
            # Split into speaker turns
            turns = [t.strip() for t in transcript_text.strip().split("\n") if t.strip()]
            
            # Render outer container
            st.markdown(
                '<div style="background:#0F172A; border:1px solid #334155; border-radius:10px; padding:12px 14px 6px 14px; max-height:420px; overflow-y:auto; font-size:0.85rem;">',
                unsafe_allow_html=True
            )
            
            # Render each turn individually (much more reliable than one giant HTML blob)
            for idx, turn in enumerate(turns):
                lower = turn.lower()
                if lower.startswith("sales rep:"):
                    label, content = "Sales Rep", turn.split(":", 1)[1].strip() if ":" in turn else ""
                    label_html = f'<span style="color:#14B8A6; font-weight:700;">{label}:</span>'
                elif lower.startswith("customer:"):
                    label, content = "Customer", turn.split(":", 1)[1].strip() if ":" in turn else ""
                    label_html = f'<span style="color:#60A5FA; font-weight:700;">{label}:</span>'
                else:
                    label_html = ""
                    content = turn
                
                is_last = (idx == len(turns) - 1)
                # No bottom border + less bottom margin on the final turn
                extra = "" if is_last else 'border-bottom: 1px solid #334155; padding-bottom: 9px;'
                
                turn_html = f'''
                <div style="margin-bottom: 8px; {extra} line-height: 1.45;">
                    {label_html} <span style="color:#E2E8F0;">{content}</span>
                </div>
                '''
                st.markdown(turn_html, unsafe_allow_html=True)
            
            # Close the container
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("No structured transcript available.")
    
    # === AGENT OUTPUTS ===
    st.markdown("### Agent Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="premium-card"><div class="premium-card-title">🤖 DealIntelAgent</div></div>', unsafe_allow_html=True)
        
        # Visual Deal Intelligence
        if deal_data:
            for key, value in deal_data.items():
                if value and str(value).strip():
                    clean_value = str(value).strip().strip('"')
                    st.markdown(f"""
                    <div style="background:#0F172A; border:1px solid #334155; border-radius:8px; padding:8px 12px; margin-bottom:6px;">
                        <div style="font-size:0.7rem; color:#64748B; text-transform:uppercase; letter-spacing:0.5px;">{key.replace('_', ' ').title()}</div>
                        <div style="color:#E2E8F0; font-size:0.9rem; margin-top:2px;">{clean_value}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.json(state.get("deal_intel", {}))
    
    with col2:
        objections_raw = state.get("objections", "{}")
        try:
            if isinstance(objections_raw, str):
                obj_dict = json.loads(objections_raw)
            else:
                obj_dict = objections_raw
            obj_list = obj_dict.get("objections", []) if isinstance(obj_dict, dict) else []
        except Exception:
            obj_list = []
        
        if obj_list:
            objections_items = ""
            for o in obj_list:
                objections_items += f'<div style="background:#0F172A; border:1px solid #334155; border-radius:6px; padding:8px 10px; margin-bottom:6px; font-size:0.85rem; color:#E2E8F0; line-height:1.35;">• {o}</div>'
        else:
            objections_items = '<div style="color:#94A3B8; font-size:0.85rem; padding:4px 0;">No objections mapped.</div>'
        
        st.markdown(f"""
        <div class="premium-card" style="padding: 1rem 1.25rem;">
            <div class="premium-card-title" style="margin-bottom: 0.6rem;">🤖 ObjectionMapperAgent</div>
            <div style="background:#0F172A; border:1px solid #334155; border-radius:8px; padding:10px; max-height:380px; overflow:auto;">
                {objections_items}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="premium-card"><div class="premium-card-title">✉️ Proposal</div></div>', unsafe_allow_html=True)
        
        proposal_text = state["proposal"]
        st.text_area(
            "Follow-up Proposal", 
            proposal_text, 
            height=420,
            key="proposal_text",
            label_visibility="collapsed"
        )
        
        # Copy to clipboard button
        if st.button("📋 Copy Proposal to Clipboard", key="copy_proposal"):
            st.toast("Proposal copied to clipboard! (Use Ctrl/Cmd + C to copy from the text area above)", icon="✅")
    
    # === FINAL SUMMARY ===
    st.markdown("### 📋 Final Summary")
    st.write(state["final_summary"])
    
    # === SHARE THIS ANALYSIS ===
    st.markdown("---")
    st.markdown("#### 📤 Share this Analysis")
    
    col_share1, col_share2 = st.columns(2)
    
    with col_share1:
        # Build a complete, well-formatted analysis for reliable download (no clipboard hacks, no toasts, no dropdowns)
        transcript_text = state.get("structured_transcript", "") or ""
        # Recompute key moments for inclusion
        highlighted = []
        if raw_transcript:
            important_keywords = ["pain", "problem", "issue", "frustrat", "expensive", "slow", "nobody", "follow up", "cost", "budget"]
            for line in raw_transcript.split("\n"):
                if any(kw in line.lower() for kw in important_keywords):
                    highlighted.append(line.strip())
        key_moments_text = "\n".join(f"- {h}" for h in highlighted) if highlighted else "None identified in this call."
        # Speaker talk time summary
        if 'speaker_times' in locals() and speaker_times:
            speaker_text = "\n".join(f"{speaker}: {count} words" for speaker, count in speaker_times.items())
        else:
            speaker_text = "Not available."
        # Nicely formatted DealIntel (key: value lines)
        if deal_data:
            deal_lines = []
            for k, v in deal_data.items():
                if v and str(v).strip() and str(v).strip() != '""':
                    clean = str(v).strip().strip('"')
                    deal_lines.append(f"{k.replace('_', ' ').title()}: {clean}")
            deal_intel_text = "\n".join(deal_lines) if deal_lines else str(state.get("deal_intel", ""))
        else:
            deal_intel_text = str(state.get("deal_intel", ""))
        # Nicely formatted Objections as bullet list
        try:
            obj_raw = state.get("objections", "{}")
            if isinstance(obj_raw, str):
                obj_dict = json.loads(obj_raw)
            else:
                obj_dict = obj_raw or {}
            objs = obj_dict.get("objections", []) if isinstance(obj_dict, dict) else []
            objections_text = "\n".join(f"- {o}" for o in objs) if objs else "No objections mapped."
        except Exception:
            objections_text = str(state.get("objections", ""))
        # Next best action (matches the insights card)
        next_action = "Send proposal & schedule follow-up" if num_objections < 5 else "Address objections first"
        # Proposal
        proposal_text = state.get("proposal", "")
    
        full_analysis_text = f"""Deal War Room - Full Analysis Report
    
Call Quality Score: {call_quality_score}/100
    
=== INSIGHTS ===
Objections Found: {num_objections}
Deal Signals Captured: {filled_fields}
Next Best Action: {next_action}
    
=== KEY MOMENTS / POTENTIAL PAIN POINTS ===
{key_moments_text}
    
=== SPEAKER TALK TIME DISTRIBUTION ===
{speaker_text}
    
=== STRUCTURED TRANSCRIPT (Speaker Labeled) ===
{transcript_text}
    
=== DEALINTELAGENT OUTPUT ===
{deal_intel_text}
    
=== OBJECTIONMAPPERAGENT OUTPUT ===
{objections_text}
    
=== FOLLOW-UP PROPOSAL ===
{proposal_text}
    
---
"""
    
        st.download_button(
            "Download Full Analysis as Text",
            data=full_analysis_text,
            file_name="closingforce_deal_analysis.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_share2:
        st.caption("Download the complete report for your records, team, or CRM.")
