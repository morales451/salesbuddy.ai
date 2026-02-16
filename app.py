"""
SalesBuddy.ai — Sales & Meeting Intelligence for Henry Company
Transcribes and analyzes sales calls and internal meetings using OpenAI.
"""

import io
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WHISPER_MAX_BYTES = 25 * 1024 * 1024  # 25 MiB — Whisper API hard limit
CHUNK_DURATION_MS = 10 * 60 * 1000  # 10-minute chunks when splitting
SUPPORTED_FORMATS = ["wav", "mp3", "m4a"]

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------
EXTERNAL_SALES_CALL_PROMPT = """\
You are an expert sales analyst and coach for a manufacturer sales representative \
at Henry Company, a leading provider of Roofing & Building Envelope Systems. \
The rep covers the Texas territory.

## Context
- Henry Company products include: Silicone coatings, Acrylic coatings, TPO roofing, \
Mod-Bit (modified bitumen), Restoration systems, Warranty programs.
- Specific product names to listen for: Pro-Grade 988, Tropi-Cool, Air-Bloc, \
Dry-Barrier, Sealant 925.
- Industry terms you may encounter: Spec (specification), Squares (roofing \
measurement = 100 sq ft), QXO (distributor / industry acquisition entity).
- The audio was recorded on a Plaud Note device. It may be mono (phone recording) \
with background noise or low fidelity.
- The rep works with contractors, distributors (including QXO), architects, and \
building owners.

## Your Task
Analyze the following sales call transcript and produce TWO clearly separated sections.

---

### SECTION 1: SALESFORCE ACTIVITY LOG

Generate a structured Salesforce call log with these exact fields:

- **Subject:** A concise, professional subject line (max 80 characters).
- **Account / Client Name:** The company or person the rep spoke with. \
If not identifiable, write "Unknown — confirm with rep."
- **Relationship Status:** One of: New Prospect | Active Opportunity | \
Existing Customer | At Risk | Unknown.
- **Products Discussed:** List all Henry Company products or product categories \
mentioned. If none were explicitly mentioned, write "None identified."
- **Next Steps:** Bullet-pointed action items committed to during the call. \
Include owners and deadlines when mentioned.

---

### SECTION 2: SALES COACH ANALYSIS

Evaluate the rep's performance against these frameworks:

**SPIN Selling Assessment:**
- Situation questions asked (understanding the customer's current state)
- Problem questions asked (uncovering pain points)
- Implication questions asked (exploring consequences of inaction)
- Need-payoff questions asked (connecting to Henry Company solutions)
- Overall SPIN Score: Weak | Developing | Strong
- Specific suggestions for improvement

**Challenger Sale Assessment:**
- Teach: Did the rep share unique insights or educate the customer?
- Tailor: Did the rep customize the message to this stakeholder's priorities?
- Take Control: Did the rep confidently guide the conversation toward next steps?
- Overall Challenger Score: Weak | Developing | Strong

**Win / Loss Indicator:**
- Deal Trajectory: Trending Win | Trending Loss | Neutral | Too Early to Tell
- Key risk factors identified
- Recommended actions to improve win probability

Be specific. Quote brief phrases from the transcript to support your assessments. \
If the transcript is too short or unclear to evaluate a dimension, say so honestly \
rather than guessing."""

INTERNAL_MEETING_PROMPT = """\
You are an expert meeting analyst and executive assistant for the sales organization \
at Henry Company, a leading provider of Roofing & Building Envelope Systems.

## Context
- This is an internal meeting (not a customer-facing call).
- Participants may include sales reps, sales managers, marketing, technical services, \
operations, and leadership.
- Henry Company products include: Silicone coatings, Acrylic coatings, TPO roofing, \
Mod-Bit (modified bitumen), Restoration systems, Warranty programs.
- Industry terms: Spec (specification), Squares (100 sq ft), QXO (distributor / \
industry acquisition entity).
- Audio may be from a Plaud Note device (may have background noise, overlapping speakers).

## Your Task
Analyze the following meeting transcript and produce THREE clearly separated sections.

---

### SECTION 1: MEETING MINUTES
- Provide a concise executive summary (3–5 sentences).
- List the key topics discussed, organized by theme.
- Note any significant data points, metrics, or figures mentioned.
- If you can identify participants by name or role, note who contributed what.

---

### SECTION 2: ACTION ITEMS
For each action item identified, provide:
- **Action:** What needs to be done (specific and actionable).
- **Owner:** Who is responsible (name or role, if identifiable).
- **Deadline:** When it is due (if mentioned; otherwise "TBD").
- **Priority:** High | Medium | Low (inferred from discussion urgency).

Format as a numbered list. If no clear action items were discussed, state that explicitly.

---

### SECTION 3: DECISIONS MADE
For each decision:
- **Decision:** What was decided.
- **Rationale:** Why (if discussed).
- **Impact:** Who or what is affected.

If no formal decisions were made, note any consensus points or directional agreements.

Be thorough but concise. If parts of the audio are unclear, note that rather than guessing."""


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_api_key():
    """Resolve the OpenAI API key from available sources."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv("OPENAI_API_KEY")


def chunk_audio_if_needed(uploaded_file):
    """Return a list of (filename, BytesIO) tuples ready for the Whisper API.

    If the file is under 25 MB it is returned as-is. Otherwise it is split
    into 10-minute chunks exported as MP3 to guarantee each chunk stays
    under the size limit.
    """
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if len(file_bytes) <= WHISPER_MAX_BYTES:
        buf = io.BytesIO(file_bytes)
        buf.name = uploaded_file.name
        return [(uploaded_file.name, buf)]

    # File exceeds limit — split with pydub
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    base_name = os.path.splitext(uploaded_file.name)[0]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        audio = AudioSegment.from_file(tmp_path)
        chunks = []
        for i in range(0, len(audio), CHUNK_DURATION_MS):
            segment = audio[i : i + CHUNK_DURATION_MS]
            buf = io.BytesIO()
            segment.export(buf, format="mp3")
            buf.seek(0)
            part_num = (i // CHUNK_DURATION_MS) + 1
            chunk_name = f"{base_name}_part{part_num}.mp3"
            buf.name = chunk_name
            chunks.append((chunk_name, buf))
        return chunks
    finally:
        os.unlink(tmp_path)


def transcribe_audio(client, file_name, file_bytes_io):
    """Transcribe a single audio buffer with Whisper and return the text."""
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=(file_name, file_bytes_io),
        response_format="text",
    )
    return transcript


def transcribe_uploaded_file(client, uploaded_file):
    """Transcribe an uploaded file, chunking if necessary.

    Returns (full_transcript, num_chunks).
    """
    chunks = chunk_audio_if_needed(uploaded_file)
    transcripts = []
    for name, buf in chunks:
        text = transcribe_audio(client, name, buf)
        transcripts.append(text.strip())
    return " ".join(transcripts), len(chunks)


def analyze_transcript(client, transcript, mode):
    """Analyze a transcript with GPT-4o using the mode-specific prompt."""
    system_prompt = (
        EXTERNAL_SALES_CALL_PROMPT
        if mode == "External Sales Call"
        else INTERNAL_MEETING_PROMPT
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here is the transcript to analyze:\n\n{transcript}",
            },
        ],
        temperature=0.3,
        max_tokens=4000,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="SalesBuddy.ai",
        page_icon="\U0001F3AF",  # dart / target emoji
        layout="wide",
    )

    st.title("SalesBuddy.ai")
    st.caption("Sales & Meeting Intelligence for Henry Company")

    # --- API key -----------------------------------------------------------
    api_key = get_api_key()
    if not api_key:
        st.error(
            "**OpenAI API key not found.** "
            "Please configure it in one of the following ways:"
        )
        st.info(
            "**Streamlit Community Cloud:** Add `OPENAI_API_KEY` in your app's "
            "*Settings > Secrets* panel.\n\n"
            "**Local development:** Create `.streamlit/secrets.toml` with:\n"
            "```\nOPENAI_API_KEY = \"sk-...\"\n```\n"
            "Or set the `OPENAI_API_KEY` environment variable."
        )
        st.stop()

    client = OpenAI(api_key=api_key)

    # --- Session state -----------------------------------------------------
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # --- Sidebar: upload & config ------------------------------------------
    with st.sidebar:
        st.header("Upload & Configure")

        uploaded_files = st.file_uploader(
            "Upload audio files",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help="Supports WAV, MP3, and M4A. Files over 25 MB are automatically split.",
        )

        mode = st.radio(
            "Processing Mode",
            options=["External Sales Call", "Internal Meeting"],
            help="Choose the analysis type to apply to all uploaded files.",
        )

        process_btn = st.button(
            "Process Files",
            type="primary",
            disabled=(not uploaded_files) or st.session_state.processing,
            use_container_width=True,
        )

        st.divider()
        st.caption("SalesBuddy.ai v1.0 | Powered by OpenAI")

    # --- Kick off processing -----------------------------------------------
    if process_btn and uploaded_files:
        st.session_state.processing = True
        st.session_state.results = {}

        total = len(uploaded_files)
        progress = st.progress(0, text="Starting batch processing\u2026")

        for idx, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            progress.progress(
                idx / total,
                text=f"Processing {idx + 1} of {total}: {file_name}",
            )

            with st.status(f"Processing: {file_name}", expanded=True) as status:
                # Step 1 — Transcribe
                st.write("Transcribing audio\u2026")
                try:
                    transcript, num_chunks = transcribe_uploaded_file(
                        client, uploaded_file
                    )
                    if num_chunks > 1:
                        st.write(
                            f"File was split into {num_chunks} chunks for processing."
                        )
                    st.write("Transcription complete.")
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")
                    st.session_state.results[file_name] = {
                        "error": f"Transcription failed: {exc}",
                    }
                    status.update(label=f"Failed: {file_name}", state="error")
                    continue

                # Step 2 — Analyze
                st.write("Analyzing with GPT-4o\u2026")
                try:
                    analysis = analyze_transcript(client, transcript, mode)
                    st.write("Analysis complete.")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.session_state.results[file_name] = {
                        "transcript": transcript,
                        "error": f"Analysis failed: {exc}",
                    }
                    status.update(label=f"Failed: {file_name}", state="error")
                    continue

                # Store result
                st.session_state.results[file_name] = {
                    "transcript": transcript,
                    "analysis": analysis,
                    "mode": mode,
                }
                status.update(label=f"Completed: {file_name}", state="complete")

        progress.progress(1.0, text="Batch processing complete.")
        st.session_state.processing = False

    # --- Display results ---------------------------------------------------
    if st.session_state.results:
        st.header("Results")

        for file_name, result in st.session_state.results.items():
            with st.expander(f"\U0001F4C4 {file_name}", expanded=True):
                if "error" in result and "transcript" not in result:
                    st.error(result["error"])
                    continue

                if "error" in result:
                    st.warning(result["error"])

                tab_analysis, tab_transcript = st.tabs(
                    ["Analysis", "Transcript"]
                )

                with tab_analysis:
                    if "analysis" in result:
                        st.markdown(result["analysis"])
                    else:
                        st.info("Analysis not available.")

                with tab_transcript:
                    st.text_area(
                        "Raw Transcript",
                        value=result.get("transcript", ""),
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                    )


if __name__ == "__main__":
    main()
