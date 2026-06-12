# ClosingForce

**AI-powered sales call analysis** — 4 specialized agents that turn a recorded sales call into structured deal intelligence, objection map, and a professional follow-up proposal.

- Local transcription (mlx-whisper)
- Gemini Flash agents (fast, high quality, free tier friendly)
- Simple LangGraph orchestration
- Streamlit "Deal War Room" UI

## Quick Start

### 1. Get a free Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key (free tier)
3. Copy it

### 2. Setup

```bash
# Clone the repo
git clone <your-repo>
cd closingforce

# (Recommended) Create & activate venv
python -m venv venv
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

### 3. Run the Deal War Room

```bash
streamlit run streamlit_app.py
```

Upload a `.wav` sales call recording and click **Run Full Workflow**.

### 4. (Optional) Quick test without UI

```bash
python test_phase1.py
```

(Requires `data/test_call.wav` to exist.)

## Environment Variables

See `.env.example` for the full template.

| Variable          | Required | Default              | Description |
|-------------------|----------|----------------------|-----------|
| `GEMINI_API_KEY`  | Yes      | —                    | Your Google Gemini API key |
| `GEMINI_MODEL`    | No       | `gemini-2.5-flash-lite` | Gemini model to use. `gemini-2.5-flash-lite` is the best current free tier choice (budget-friendly + higher quotas). |
| `TEMPERATURE`     | No       | `0.0`                | LLM creativity (0.0 = deterministic) |

**Free tier notes**: Gemini free tier quotas are tight and reset daily at midnight Pacific Time. We default to `gemini-2.5-flash-lite` (most budget-friendly) and include automatic retry with backoff for 429 errors. Always monitor your real usage here: https://aistudio.google.com/rate-limit

## Architecture (Current)

```
Audio (.wav)
   ↓ (mlx-whisper - local)
Raw Transcript
   ↓
SpeakerLabelerAgent → Structured Transcript
   ↓
DealIntelAgent      → MEDDPICC / deal intel (JSON-ish)
   ↓
ObjectionMapperAgent → Customer objections list
   ↓
ProposalCrafterAgent → Professional follow-up email/proposal
   ↓
Final Summary (combined)
```

Built with:
- **LangGraph** for the simple 5-node sequential workflow
- **LangChain** prompt + LLM chains
- No tools, no memory, no structured_output yet (plain text/JSON in prompts)

## Switching Models

Change `GEMINI_MODEL` in `.env` to any Gemini model your key has access to:

**Recommended for free tier**:
- `gemini-2.5-flash-lite` (default — fastest + most budget-friendly, best chance of staying under quota)
- `gemini-2.5-flash` (higher quality when you have quota headroom)

**Older models**:
- Avoid older Flash models (pre-2.5) for new work — they are deprecated by Google.

## Migration from Ollama (Previous Setup)

Previously used `langchain-ollama` + `OLLAMA_MODEL`.

To switch back:
1. Re-add `langchain-ollama` to requirements
2. Change the four agent files back to `ChatOllama`
3. Restore `OLLAMA_MODEL` in `.env`

## Development

- All agent logic lives in `agents/*.py` (each file creates its own `llm` instance for simplicity)
- Graph wiring is in `agents/graph.py`
- UI is a single-file Streamlit app

## Troubleshooting

**"API key not valid"**  
→ Double-check `GEMINI_API_KEY` in `.env` (no extra spaces/quotes).

**Rate limit / quota errors (429 RESOURCE_EXHAUSTED)**  (free tier resets daily at midnight PT; app now has automatic retry + backoff)  
→ Free tier has RPM/RPD limits. Wait or use a lighter model / lower temperature. Consider upgrading if doing heavy testing.

**ImportError for langchain_google_genai**  
→ `pip install langchain-google-genai` (or re-run `pip install -r requirements.txt`)

**Gemini returns markdown instead of clean JSON**  
→ The prompts instruct "Return ONLY JSON". Gemini Flash usually follows this well; if not, you can add output parsers later.

## Future / Phase 2 Ideas (not implemented)

- Real tool calling + LangGraph tools
- Vector memory of past deals (Chroma already present)
- Human-in-the-loop approvals
- More agents (FollowUp, CRM sync, etc.)
- Tracing (LangSmith / local alternatives)

---

Built as a focused, working 4-agent Phase 1 implementation. The original vision was a full 6-agent autonomous revenue engine.
