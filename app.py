"""
SalesBuddy.ai — Sales & Meeting Intelligence for Henry Company
Transcribes and analyzes sales calls and internal meetings using OpenAI.

v2.0 Features:
- Per-file mode selection (External Sales Call / Internal Meeting)
- Competitive intelligence extraction
- Objection handling analysis
- Visual coaching scorecard (SPIN, Challenger, Deal Trajectory)
- Copy-to-clipboard for Salesforce log
- Download individual & combined reports
- CSV export of Salesforce fields
- End-of-day pipeline summary
- Follow-up email drafts
- Salesforce API integration (optional)
"""

import csv
import io
import os
import re
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

SCORE_MAP = {"strong": 1.0, "developing": 0.5, "weak": 0.15}
TRAJECTORY_CONFIG = {
    "trending win": ("\u2705", "green"),
    "trending loss": ("\u274C", "red"),
    "neutral": ("\u2796", "orange"),
    "too early to tell": ("\u2753", "gray"),
}

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
Analyze the following sales call transcript and produce FOUR clearly separated sections.

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

---

### SECTION 3: COMPETITIVE INTELLIGENCE

Analyze the transcript for any competitive mentions or market intelligence:

- **Competitors Mentioned:** List any competitor companies, brands, or products \
referenced (e.g., GAF, Carlisle, Firestone/Holcim, Johns Manville, Tremco, \
Polyglass, Mule-Hide, Soprema). If none, write "None identified."
- **Competitive Context:** For each competitor mention, note the context — \
was the customer comparing products, mentioning an existing install, \
considering switching, etc.?
- **Customer Sentiment on Competitors:** Positive, Negative, or Neutral toward \
each competitor mentioned.
- **Switching Signals:** Any indications the customer is considering moving away \
from Henry Company to a competitor, or from a competitor to Henry Company.
- **Market Intelligence:** Any broader market trends, pricing intel, or industry \
developments mentioned during the call.

---

### SECTION 4: OBJECTION HANDLING ANALYSIS

Identify and evaluate how the rep handled customer objections:

- For each objection identified, provide:
  - **Objection:** What the customer pushed back on (price, timing, product fit, etc.)
  - **Rep's Response:** How the rep addressed it (briefly summarize).
  - **Effectiveness:** Effective | Partially Effective | Ineffective
  - **Suggested Alternative:** A stronger response the rep could have used.

- If no objections were raised, note that explicitly and suggest probing questions \
the rep could use to surface hidden objections.

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
- Provide a concise executive summary (3\u20135 sentences).
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

PIPELINE_SUMMARY_PROMPT = """\
You are a sales operations analyst for Henry Company (Roofing & Building Envelope Systems).

Below are the individual analysis reports from today's batch of sales calls and meetings. \
Synthesize them into a concise end-of-day pipeline briefing.

## Output Format

### DAILY PIPELINE SUMMARY

**Overview:**
- Total calls/meetings processed: [count]
- External sales calls: [count]
- Internal meetings: [count]

**Accounts Touched Today:**
For each account, provide a one-line status (relationship status + deal trajectory if applicable).

**Aggregate Next Steps (Priority Order):**
Consolidate all action items across calls, de-duplicate, and prioritize. For each:
- Action item
- Account/context
- Owner (if identified)
- Urgency: High | Medium | Low

**Pipeline Health Snapshot:**
- Deals trending positive
- Deals at risk (with reason)
- New opportunities identified

**Competitive Landscape:**
Summarize any competitor mentions across all calls.

**Top Priorities for Tomorrow:**
Based on all calls, recommend the rep's top 3 priorities for the next business day.

Be direct and actionable. This is for a busy sales rep reviewing their day in 2 minutes."""

FOLLOW_UP_EMAIL_PROMPT = """\
You are a professional sales communication writer for Henry Company \
(Roofing & Building Envelope Systems).

Based on the call analysis below, draft a professional follow-up email \
from the sales rep to the customer.

## Guidelines:
- Professional but warm tone
- Reference specific topics discussed (products, projects, needs)
- Confirm any agreed-upon next steps with dates if mentioned
- Include a clear call to action
- Keep it concise (under 200 words)
- Do NOT include a subject line — just the email body
- Sign off with just a placeholder: [Your Name]

## Call Analysis:
"""


# ---------------------------------------------------------------------------
# Helper Functions — Audio
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
    """Transcribe an uploaded file, chunking if necessary."""
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
# Helper Functions — Section Parsing
# ---------------------------------------------------------------------------
def extract_section(analysis, section_keyword):
    """Extract a section from the markdown analysis by keyword in its header."""
    pattern = rf"(###\s+.*?{re.escape(section_keyword)}.*?\n)(.*?)(?=\n###\s|\Z)"
    match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
    if match:
        return (match.group(1) + match.group(2)).strip()
    return None


def extract_salesforce_block(analysis):
    """Extract the Salesforce Activity Log section."""
    return extract_section(analysis, "SALESFORCE ACTIVITY LOG")


def parse_salesforce_fields(analysis):
    """Parse structured fields from the Salesforce Activity Log section."""
    sf_section = extract_salesforce_block(analysis) or ""
    fields = {}

    field_patterns = {
        "Subject": r"\*\*Subject:\*\*\s*(.+?)(?:\n|$)",
        "Account": r"\*\*Account\s*/\s*Client Name:\*\*\s*(.+?)(?:\n|$)",
        "Relationship Status": r"\*\*Relationship Status:\*\*\s*(.+?)(?:\n|$)",
        "Products Discussed": r"\*\*Products Discussed:\*\*\s*(.+?)(?:\n###|\n\*\*Next|\Z)",
        "Next Steps": r"\*\*Next Steps:\*\*\s*(.+?)(?:\n###|\Z)",
    }

    for field, pattern in field_patterns.items():
        match = re.search(pattern, sf_section, re.DOTALL | re.IGNORECASE)
        fields[field] = match.group(1).strip() if match else ""

    return fields


def parse_scores(analysis):
    """Parse SPIN, Challenger, and Deal Trajectory scores from analysis."""
    scores = {}

    spin = re.search(
        r"Overall SPIN Score:\s*(Weak|Developing|Strong)", analysis, re.IGNORECASE
    )
    if spin:
        scores["spin"] = spin.group(1).strip().lower()

    challenger = re.search(
        r"Overall Challenger Score:\s*(Weak|Developing|Strong)",
        analysis,
        re.IGNORECASE,
    )
    if challenger:
        scores["challenger"] = challenger.group(1).strip().lower()

    trajectory = re.search(
        r"Deal Trajectory:\s*(Trending Win|Trending Loss|Neutral|Too Early to Tell)",
        analysis,
        re.IGNORECASE,
    )
    if trajectory:
        scores["trajectory"] = trajectory.group(1).strip().lower()

    return scores


# ---------------------------------------------------------------------------
# Helper Functions — Export
# ---------------------------------------------------------------------------
def results_to_markdown(results):
    """Convert all results to a single markdown document."""
    parts = ["# SalesBuddy.ai \u2014 Analysis Report\n"]
    for file_name, result in results.items():
        parts.append(f"\n---\n\n## {file_name}\n")
        parts.append(f"**Mode:** {result.get('mode', 'N/A')}\n")
        if "analysis" in result:
            parts.append(f"\n{result['analysis']}\n")
        if "transcript" in result:
            parts.append(f"\n### Raw Transcript\n\n{result['transcript']}\n")
        if "error" in result:
            parts.append(f"\n> **Error:** {result['error']}\n")
    return "\n".join(parts)


def results_to_csv(results):
    """Export Salesforce fields from all external call results as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "File",
        "Subject",
        "Account / Client Name",
        "Relationship Status",
        "Products Discussed",
        "Next Steps",
    ])

    has_rows = False
    for file_name, result in results.items():
        if result.get("mode") != "External Sales Call":
            continue
        if "analysis" not in result:
            continue
        fields = parse_salesforce_fields(result["analysis"])
        writer.writerow([
            file_name,
            fields.get("Subject", ""),
            fields.get("Account", ""),
            fields.get("Relationship Status", ""),
            fields.get("Products Discussed", ""),
            fields.get("Next Steps", ""),
        ])
        has_rows = True

    return output.getvalue() if has_rows else ""


# ---------------------------------------------------------------------------
# Helper Functions — AI Generation
# ---------------------------------------------------------------------------
def generate_pipeline_summary(client, results):
    """Generate an end-of-day pipeline summary across all processed calls."""
    combined = []
    for file_name, result in results.items():
        if "analysis" in result:
            combined.append(
                f"## {file_name} ({result.get('mode', 'Unknown')})\n\n"
                f"{result['analysis']}"
            )

    if not combined:
        return "No analyses available to summarize."

    all_analyses = "\n\n---\n\n".join(combined)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PIPELINE_SUMMARY_PROMPT},
            {"role": "user", "content": all_analyses},
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    return response.choices[0].message.content


def generate_follow_up_email(client, analysis):
    """Generate a follow-up email based on a call analysis."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": FOLLOW_UP_EMAIL_PROMPT},
            {"role": "user", "content": analysis},
        ],
        temperature=0.4,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Helper Functions — Salesforce Integration
# ---------------------------------------------------------------------------
def get_salesforce_client():
    """Create a Salesforce client from configured credentials.

    Returns (client, error_message). On success error_message is None.
    """
    try:
        from simple_salesforce import Salesforce
    except ImportError:
        return None, (
            "`simple-salesforce` is not installed. "
            "Run `pip install simple-salesforce` to enable this feature."
        )

    sf_user = st.session_state.get("sf_username", "")
    sf_pass = st.session_state.get("sf_password", "")
    sf_token = st.session_state.get("sf_security_token", "")
    sf_domain = st.session_state.get("sf_domain", "login")

    if not all([sf_user, sf_pass, sf_token]):
        return None, (
            "Salesforce credentials not configured. "
            "Fill in the Salesforce Integration section in the sidebar."
        )

    try:
        sf = Salesforce(
            username=sf_user,
            password=sf_pass,
            security_token=sf_token,
            domain=sf_domain,
        )
        return sf, None
    except Exception as e:
        return None, f"Salesforce connection failed: {e}"


def push_to_salesforce(sf, fields, full_analysis):
    """Create a Task record in Salesforce with the call log data."""
    task_data = {
        "Subject": fields.get("Subject", "Sales Call — SalesBuddy.ai")[:255],
        "Description": full_analysis[:32000],
        "Status": "Completed",
        "Priority": "Normal",
        "Type": "Call",
    }
    return sf.Task.create(task_data)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_coaching_scorecard(scores):
    """Render visual coaching scorecard from parsed scores."""
    if not scores:
        return

    st.markdown("#### Coaching Scorecard")
    cols = st.columns(3)

    with cols[0]:
        if "spin" in scores:
            val = SCORE_MAP.get(scores["spin"], 0)
            label = scores["spin"].title()
            st.metric("SPIN Selling", label)
            st.progress(val)

    with cols[1]:
        if "challenger" in scores:
            val = SCORE_MAP.get(scores["challenger"], 0)
            label = scores["challenger"].title()
            st.metric("Challenger Sale", label)
            st.progress(val)

    with cols[2]:
        if "trajectory" in scores:
            traj = scores["trajectory"]
            icon, color = TRAJECTORY_CONFIG.get(traj, ("\u2753", "gray"))
            label = traj.title()
            st.metric("Deal Trajectory", f"{icon} {label}")


def render_copy_section(text, label, key):
    """Render a collapsible copy-friendly code block."""
    with st.expander(f"Copy: {label}"):
        st.code(text, language=None)


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="SalesBuddy.ai",
        page_icon="\U0001F3AF",
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
    for key, default in {
        "results": {},
        "processing": False,
        "pipeline_summary": None,
        "follow_up_emails": {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.header("Upload & Configure")

        uploaded_files = st.file_uploader(
            "Upload audio files",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help="Supports WAV, MP3, and M4A. Files over 25 MB are automatically split.",
        )

        # Default mode + per-file overrides
        default_mode = st.radio(
            "Default Processing Mode",
            options=["External Sales Call", "Internal Meeting"],
            help="Applied to all files. Override individual files below.",
        )

        if uploaded_files and len(uploaded_files) > 1:
            with st.expander("Per-file mode overrides"):
                for uf in uploaded_files:
                    st.selectbox(
                        uf.name,
                        options=["External Sales Call", "Internal Meeting"],
                        index=(
                            0 if default_mode == "External Sales Call" else 1
                        ),
                        key=f"mode_{uf.name}",
                    )

        process_btn = st.button(
            "Process Files",
            type="primary",
            disabled=(not uploaded_files) or st.session_state.processing,
            use_container_width=True,
        )

        st.divider()

        # Salesforce Integration Config
        with st.expander("Salesforce Integration (Optional)"):
            st.caption(
                "Configure credentials to push call logs directly to Salesforce."
            )
            st.text_input("Username", key="sf_username")
            st.text_input("Password", key="sf_password", type="password")
            st.text_input("Security Token", key="sf_security_token", type="password")
            st.selectbox(
                "Domain",
                options=["login", "test"],
                key="sf_domain",
                help="Use 'test' for sandbox environments.",
            )

        st.divider()
        st.caption("SalesBuddy.ai v2.0 | Powered by OpenAI")

    # --- Helper to resolve per-file mode -----------------------------------
    def get_file_mode(file_name):
        return st.session_state.get(f"mode_{file_name}", default_mode)

    # --- Kick off processing -----------------------------------------------
    if process_btn and uploaded_files:
        st.session_state.processing = True
        st.session_state.results = {}
        st.session_state.pipeline_summary = None
        st.session_state.follow_up_emails = {}

        total = len(uploaded_files)
        progress = st.progress(0, text="Starting batch processing\u2026")

        for idx, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            mode = get_file_mode(file_name)
            progress.progress(
                idx / total,
                text=f"Processing {idx + 1} of {total}: {file_name}",
            )

            with st.status(
                f"Processing: {file_name} [{mode}]", expanded=True
            ) as status:
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

                st.session_state.results[file_name] = {
                    "transcript": transcript,
                    "analysis": analysis,
                    "mode": mode,
                }
                status.update(label=f"Completed: {file_name}", state="complete")

        progress.progress(1.0, text="Batch processing complete.")
        st.session_state.processing = False

    # --- Display results ---------------------------------------------------
    if not st.session_state.results:
        return

    st.header("Results")

    # --- Global action bar -------------------------------------------------
    action_cols = st.columns([1, 1, 1, 1])

    with action_cols[0]:
        full_report = results_to_markdown(st.session_state.results)
        st.download_button(
            "\u2B07 Download Full Report",
            data=full_report,
            file_name="salesbuddy_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with action_cols[1]:
        csv_data = results_to_csv(st.session_state.results)
        st.download_button(
            "\u2B07 Export Salesforce CSV",
            data=csv_data if csv_data else "No external call data to export.",
            file_name="salesforce_import.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=(not csv_data),
        )

    with action_cols[2]:
        if st.button(
            "\U0001F4CA Generate Pipeline Summary", use_container_width=True
        ):
            with st.spinner("Generating end-of-day pipeline summary\u2026"):
                st.session_state.pipeline_summary = generate_pipeline_summary(
                    client, st.session_state.results
                )
            st.rerun()

    # --- Pipeline Summary --------------------------------------------------
    if st.session_state.pipeline_summary:
        st.divider()
        st.subheader("Daily Pipeline Summary")
        st.markdown(st.session_state.pipeline_summary)
        dl_col, copy_col = st.columns(2)
        with dl_col:
            st.download_button(
                "\u2B07 Download Summary",
                data=st.session_state.pipeline_summary,
                file_name="pipeline_summary.md",
                mime="text/markdown",
            )
        with copy_col:
            render_copy_section(
                st.session_state.pipeline_summary,
                "Pipeline Summary",
                "copy_pipeline",
            )
        st.divider()

    # --- Individual results ------------------------------------------------
    for file_name, result in st.session_state.results.items():
        mode = result.get("mode", "Unknown")
        is_external = mode == "External Sales Call"

        with st.expander(f"\U0001F4C4 {file_name} \u2014 {mode}", expanded=True):
            if "error" in result and "transcript" not in result:
                st.error(result["error"])
                continue

            if "error" in result:
                st.warning(result["error"])

            # -- Visual coaching scorecard (external calls only) --
            if is_external and "analysis" in result:
                scores = parse_scores(result["analysis"])
                if scores:
                    render_coaching_scorecard(scores)
                    st.divider()

            # -- Tabs --
            if is_external:
                tab_names = [
                    "Analysis",
                    "Transcript",
                    "Salesforce Log",
                    "Competitive Intel",
                    "Objections",
                    "Follow-Up Email",
                    "Push to SF",
                ]
            else:
                tab_names = ["Analysis", "Transcript"]

            tabs = st.tabs(tab_names)

            # Tab: Analysis
            with tabs[0]:
                if "analysis" in result:
                    st.markdown(result["analysis"])
                    col_dl, col_cp = st.columns(2)
                    with col_dl:
                        st.download_button(
                            "\u2B07 Download Analysis",
                            data=result["analysis"],
                            file_name=f"{os.path.splitext(file_name)[0]}_analysis.md",
                            mime="text/markdown",
                            key=f"dl_analysis_{file_name}",
                        )
                    with col_cp:
                        render_copy_section(
                            result["analysis"],
                            "Full Analysis",
                            f"copy_analysis_{file_name}",
                        )
                else:
                    st.info("Analysis not available.")

            # Tab: Transcript
            with tabs[1]:
                st.text_area(
                    "Raw Transcript",
                    value=result.get("transcript", ""),
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"ta_transcript_{file_name}",
                )
                if result.get("transcript"):
                    st.download_button(
                        "\u2B07 Download Transcript",
                        data=result["transcript"],
                        file_name=f"{os.path.splitext(file_name)[0]}_transcript.txt",
                        mime="text/plain",
                        key=f"dl_transcript_{file_name}",
                    )

            # External-only tabs
            if is_external and "analysis" in result:
                analysis_text = result["analysis"]

                # Tab: Salesforce Log
                with tabs[2]:
                    sf_block = extract_salesforce_block(analysis_text)
                    if sf_block:
                        st.markdown(sf_block)
                        render_copy_section(
                            sf_block,
                            "Salesforce Activity Log",
                            f"copy_sf_{file_name}",
                        )
                    else:
                        st.info("Salesforce log section not found in analysis.")

                # Tab: Competitive Intel
                with tabs[3]:
                    ci_block = extract_section(analysis_text, "COMPETITIVE INTELLIGENCE")
                    if ci_block:
                        st.markdown(ci_block)
                        render_copy_section(
                            ci_block,
                            "Competitive Intel",
                            f"copy_ci_{file_name}",
                        )
                    else:
                        st.info("Competitive intelligence section not found.")

                # Tab: Objections
                with tabs[4]:
                    obj_block = extract_section(
                        analysis_text, "OBJECTION HANDLING"
                    )
                    if obj_block:
                        st.markdown(obj_block)
                        render_copy_section(
                            obj_block,
                            "Objection Analysis",
                            f"copy_obj_{file_name}",
                        )
                    else:
                        st.info("Objection handling section not found.")

                # Tab: Follow-Up Email
                with tabs[5]:
                    email_key = file_name
                    if email_key in st.session_state.follow_up_emails:
                        st.markdown("**Draft Follow-Up Email:**")
                        st.markdown(st.session_state.follow_up_emails[email_key])
                        render_copy_section(
                            st.session_state.follow_up_emails[email_key],
                            "Follow-Up Email",
                            f"copy_email_{file_name}",
                        )
                    else:
                        st.info(
                            "Click below to generate a follow-up email based on "
                            "this call's analysis."
                        )
                        if st.button(
                            "Generate Follow-Up Email",
                            key=f"gen_email_{file_name}",
                        ):
                            with st.spinner("Drafting follow-up email\u2026"):
                                email = generate_follow_up_email(
                                    client, analysis_text
                                )
                                st.session_state.follow_up_emails[email_key] = email
                            st.rerun()

                # Tab: Push to Salesforce
                with tabs[6]:
                    st.markdown(
                        "Push this call's activity log directly to Salesforce "
                        "as a completed Task record."
                    )
                    sf_block_for_push = extract_salesforce_block(analysis_text)
                    if sf_block_for_push:
                        st.markdown(sf_block_for_push)
                    st.divider()
                    if st.button(
                        "Push to Salesforce",
                        key=f"push_sf_{file_name}",
                        type="primary",
                    ):
                        sf, error = get_salesforce_client()
                        if error:
                            st.error(error)
                        else:
                            try:
                                fields = parse_salesforce_fields(analysis_text)
                                push_result = push_to_salesforce(
                                    sf, fields, analysis_text
                                )
                                if push_result.get("success"):
                                    st.success(
                                        f"Task created in Salesforce! "
                                        f"ID: {push_result['id']}"
                                    )
                                else:
                                    st.error(
                                        f"Salesforce returned: {push_result}"
                                    )
                            except Exception as e:
                                st.error(f"Failed to push to Salesforce: {e}")


if __name__ == "__main__":
    main()
