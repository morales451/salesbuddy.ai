"""
SalesBuddy.ai — Sales & Meeting Intelligence for Henry Company
Transcribes and analyzes sales calls and internal meetings using OpenAI.

v3.0 Features:
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
- Speaker diarization with talk-to-listen ratio
- Session history with SQLite persistence
- Coaching trends over time
"""

import csv
import io
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
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
SCORE_NUMERIC = {"strong": 3, "developing": 2, "weak": 1}
TRAJECTORY_CONFIG = {
    "trending win": ("\u2705", "green"),
    "trending loss": ("\u274C", "red"),
    "neutral": ("\u2796", "orange"),
    "too early to tell": ("\u2753", "gray"),
}

DATA_DIR = Path(os.environ.get("SALESBUDDY_DATA_DIR", ".salesbuddy_data"))
DB_PATH = DATA_DIR / "history.db"

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

DIARIZATION_PROMPT = """\
You are a conversation analysis expert specializing in sales call recordings.

The following is a transcript from a recorded sales call between a Henry Company \
sales representative and a customer/prospect. The recording was made on a Plaud Note \
device and may be mono audio with both speakers on one channel.

## Your Task
1. Identify speaker turns in the conversation.
2. Label the sales representative as "Rep" and the customer/prospect as "Customer". \
If you can identify speakers by name, add the name in parentheses.
3. Count the approximate words spoken by each speaker.

## Output
Return ONLY a valid JSON object with this exact structure (no markdown fencing):
{
  "diarized_transcript": "**Rep:** [text]\\n\\n**Customer:** [text]\\n\\n...",
  "speakers": {
    "Rep": {"word_count": 0, "turn_count": 0},
    "Customer": {"word_count": 0, "turn_count": 0}
  },
  "talk_ratio": 0.0
}

Rules for the JSON:
- "diarized_transcript": The full transcript reformatted with speaker labels. \
Use **Rep:** and **Customer:** prefixes. Separate turns with blank lines.
- "speakers": Word count and turn count for each speaker.
- "talk_ratio": Rep's proportion of total words as a float between 0 and 1 \
(e.g., 0.65 means the rep spoke 65% of the time).

If you cannot reliably distinguish speakers (e.g., transcript is too short or \
unclear), set talk_ratio to -1 and put the original text as diarized_transcript \
with a note at the top."""


# ---------------------------------------------------------------------------
# Helper Functions — Database
# ---------------------------------------------------------------------------
def init_database():
    """Initialize the SQLite database and create tables if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            transcript TEXT,
            diarized_transcript TEXT,
            analysis TEXT,
            spin_score TEXT,
            challenger_score TEXT,
            deal_trajectory TEXT,
            talk_ratio REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_analysis(file_name, mode, transcript, analysis, diarization_data=None):
    """Save an analysis result to the database."""
    scores = parse_scores(analysis) if analysis else {}
    talk_ratio = None
    diarized_transcript = None

    if diarization_data:
        talk_ratio = diarization_data.get("talk_ratio")
        diarized_transcript = diarization_data.get("diarized_transcript")
        if talk_ratio == -1:
            talk_ratio = None

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        INSERT INTO analyses
            (file_name, mode, transcript, diarized_transcript, analysis,
             spin_score, challenger_score, deal_trajectory, talk_ratio, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_name,
            mode,
            transcript,
            diarized_transcript,
            analysis,
            scores.get("spin"),
            scores.get("challenger"),
            scores.get("trajectory"),
            talk_ratio,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_history(limit=50, offset=0):
    """Retrieve analysis history from the database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, file_name, mode, transcript, diarized_transcript, analysis,
               spin_score, challenger_score, deal_trajectory, talk_ratio, created_at
        FROM analyses
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_history_count():
    """Return total number of saved analyses."""
    conn = sqlite3.connect(str(DB_PATH))
    count = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    conn.close()
    return count


def delete_history_entry(entry_id):
    """Delete a single history entry by ID."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM analyses WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()


def search_history(query, mode_filter=None, limit=50):
    """Full-text search across transcripts, analyses, and file names.

    Returns matching rows with a snippet of the matching context.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Use LIKE for broad matching — works well for small local datasets
    like_pattern = f"%{query}%"
    params = [like_pattern, like_pattern, like_pattern]

    mode_clause = ""
    if mode_filter and mode_filter != "All":
        mode_clause = "AND mode = ?"
        params.append(mode_filter)

    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT id, file_name, mode, transcript, diarized_transcript, analysis,
               spin_score, challenger_score, deal_trajectory, talk_ratio, created_at
        FROM analyses
        WHERE (
            transcript LIKE ? COLLATE NOCASE
            OR analysis LIKE ? COLLATE NOCASE
            OR file_name LIKE ? COLLATE NOCASE
        )
        {mode_clause}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_coaching_scores_over_time():
    """Retrieve coaching scores over time for trend analysis."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT file_name, spin_score, challenger_score, deal_trajectory,
               talk_ratio, created_at
        FROM analyses
        WHERE mode = 'External Sales Call'
          AND (spin_score IS NOT NULL OR challenger_score IS NOT NULL)
        ORDER BY created_at ASC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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
# Helper Functions — Speaker Diarization
# ---------------------------------------------------------------------------
def diarize_transcript(client, transcript):
    """Use GPT-4o to identify speakers and calculate talk ratio.

    Returns a dict with diarized_transcript, speakers, and talk_ratio,
    or None on failure.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": DIARIZATION_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


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
        if result.get("talk_ratio") is not None and result["talk_ratio"] >= 0:
            parts.append(
                f"**Talk Ratio (Rep):** {result['talk_ratio']:.0%}\n"
            )
        if "analysis" in result:
            parts.append(f"\n{result['analysis']}\n")
        if "diarized_transcript" in result:
            parts.append(
                f"\n### Speaker-Labeled Transcript\n\n"
                f"{result['diarized_transcript']}\n"
            )
        elif "transcript" in result:
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
        "Subject": fields.get("Subject", "Sales Call \u2014 SalesBuddy.ai")[:255],
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
            icon, _color = TRAJECTORY_CONFIG.get(traj, ("\u2753", "gray"))
            label = traj.title()
            st.metric("Deal Trajectory", f"{icon} {label}")


def render_talk_ratio(diarization_data):
    """Render talk-to-listen ratio visualization."""
    if not diarization_data:
        return

    talk_ratio = diarization_data.get("talk_ratio")
    if talk_ratio is None or talk_ratio < 0:
        return

    speakers = diarization_data.get("speakers", {})
    rep_info = speakers.get("Rep", {})
    cust_info = speakers.get("Customer", {})

    st.markdown("#### Talk-to-Listen Ratio")
    cols = st.columns([1, 1, 1])

    with cols[0]:
        st.metric("Rep Speaking", f"{talk_ratio:.0%}")
        st.progress(min(talk_ratio, 1.0))

    with cols[1]:
        listen_ratio = 1.0 - talk_ratio
        st.metric("Customer Speaking", f"{listen_ratio:.0%}")
        st.progress(min(listen_ratio, 1.0))

    with cols[2]:
        rep_words = rep_info.get("word_count", 0)
        cust_words = cust_info.get("word_count", 0)
        st.metric("Rep Words", f"{rep_words:,}")
        st.metric("Customer Words", f"{cust_words:,}")

    # Coaching tip based on ratio
    if talk_ratio > 0.70:
        st.warning(
            "The rep spoke more than 70% of the time. "
            "Best practice is 40\u201360% to allow the customer to share needs."
        )
    elif talk_ratio < 0.30:
        st.info(
            "The rep spoke less than 30% of the time. "
            "Consider whether enough value was communicated."
        )


def render_copy_section(text, label, key):
    """Render a collapsible copy-friendly code block."""
    with st.expander(f"Copy: {label}"):
        st.code(text, language=None)


# ---------------------------------------------------------------------------
# Page: Process
# ---------------------------------------------------------------------------
def page_process(client):
    """Main processing page — upload, transcribe, analyze, display."""

    # --- Session state -----------------------------------------------------
    for key, default in {
        "results": {},
        "processing": False,
        "pipeline_summary": None,
        "follow_up_emails": {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Sidebar controls --------------------------------------------------
    with st.sidebar:
        st.header("Upload & Configure")

        uploaded_files = st.file_uploader(
            "Upload audio files",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help="Supports WAV, MP3, and M4A. Files over 25 MB are automatically split.",
        )

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

        enable_diarization = st.checkbox(
            "Enable Speaker Diarization",
            value=False,
            help="Identifies Rep vs Customer, calculates talk ratio. Adds one extra API call per file (~$0.01).",
        )

        process_btn = st.button(
            "Process Files",
            type="primary",
            disabled=(not uploaded_files) or st.session_state.processing,
            use_container_width=True,
        )

        st.divider()

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
        st.caption("SalesBuddy.ai v3.0 | Powered by OpenAI")

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
                            f"File was split into {num_chunks} chunks."
                        )
                    st.write("Transcription complete.")
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")
                    st.session_state.results[file_name] = {
                        "error": f"Transcription failed: {exc}",
                    }
                    status.update(label=f"Failed: {file_name}", state="error")
                    continue

                # Step 2 — Diarization (optional)
                diarization_data = None
                if enable_diarization:
                    st.write("Identifying speakers\u2026")
                    try:
                        diarization_data = diarize_transcript(client, transcript)
                        if diarization_data:
                            st.write("Speaker diarization complete.")
                        else:
                            st.write("Diarization returned no data.")
                    except Exception as exc:
                        st.warning(f"Diarization failed (non-fatal): {exc}")

                # Step 3 — Analyze
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

                # Build result
                result_data = {
                    "transcript": transcript,
                    "analysis": analysis,
                    "mode": mode,
                }

                if diarization_data:
                    result_data["diarization"] = diarization_data
                    result_data["diarized_transcript"] = diarization_data.get(
                        "diarized_transcript", ""
                    )
                    result_data["talk_ratio"] = diarization_data.get(
                        "talk_ratio"
                    )

                st.session_state.results[file_name] = result_data

                # Save to history database
                try:
                    save_analysis(
                        file_name, mode, transcript, analysis, diarization_data
                    )
                except Exception as exc:
                    st.warning(f"Could not save to history: {exc}")

                status.update(
                    label=f"Completed: {file_name}", state="complete"
                )

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

        with st.expander(
            f"\U0001F4C4 {file_name} \u2014 {mode}", expanded=True
        ):
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

            # -- Talk ratio (if diarization was run) --
            if result.get("diarization"):
                render_talk_ratio(result["diarization"])

            if (is_external and "analysis" in result) or result.get(
                "diarization"
            ):
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
                # Show diarized transcript if available
                if result.get("diarized_transcript"):
                    st.markdown("**Speaker-Labeled Transcript:**")
                    st.markdown(result["diarized_transcript"])
                    st.divider()
                    with st.expander("Raw Transcript (unlabeled)"):
                        st.text_area(
                            "Raw",
                            value=result.get("transcript", ""),
                            height=200,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"ta_raw_{file_name}",
                        )
                else:
                    st.text_area(
                        "Raw Transcript",
                        value=result.get("transcript", ""),
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"ta_transcript_{file_name}",
                    )

                if result.get("transcript"):
                    dl_text = result.get("diarized_transcript") or result[
                        "transcript"
                    ]
                    st.download_button(
                        "\u2B07 Download Transcript",
                        data=dl_text,
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
                        st.info(
                            "Salesforce log section not found in analysis."
                        )

                # Tab: Competitive Intel
                with tabs[3]:
                    ci_block = extract_section(
                        analysis_text, "COMPETITIVE INTELLIGENCE"
                    )
                    if ci_block:
                        st.markdown(ci_block)
                        render_copy_section(
                            ci_block,
                            "Competitive Intel",
                            f"copy_ci_{file_name}",
                        )
                    else:
                        st.info(
                            "Competitive intelligence section not found."
                        )

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
                        st.markdown(
                            st.session_state.follow_up_emails[email_key]
                        )
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
                                st.session_state.follow_up_emails[
                                    email_key
                                ] = email
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
                                st.error(
                                    f"Failed to push to Salesforce: {e}"
                                )


# ---------------------------------------------------------------------------
# Page: History
# ---------------------------------------------------------------------------
def page_history():
    """Browse past analyses saved to the local database."""
    st.header("Session History")

    total = get_history_count()
    if total == 0:
        st.info(
            "No history yet. Process some audio files on the **Process** page "
            "and they will appear here automatically."
        )
        return

    st.caption(f"{total} saved analyses")

    # Pagination
    page_size = 10
    total_pages = max(1, (total + page_size - 1) // page_size)

    if "history_page" not in st.session_state:
        st.session_state.history_page = 1

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button(
            "\u25C0 Previous",
            disabled=st.session_state.history_page <= 1,
            use_container_width=True,
        ):
            st.session_state.history_page -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f"<div style='text-align:center;padding-top:8px;'>"
            f"Page {st.session_state.history_page} of {total_pages}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button(
            "Next \u25B6",
            disabled=st.session_state.history_page >= total_pages,
            use_container_width=True,
        ):
            st.session_state.history_page += 1
            st.rerun()

    offset = (st.session_state.history_page - 1) * page_size
    entries = get_history(limit=page_size, offset=offset)

    for entry in entries:
        created = entry["created_at"][:16].replace("T", " ")
        mode_tag = entry["mode"]
        label = f"{entry['file_name']} \u2014 {mode_tag} \u2014 {created}"

        with st.expander(label, expanded=False):
            # Scores summary
            score_parts = []
            if entry.get("spin_score"):
                score_parts.append(f"SPIN: {entry['spin_score'].title()}")
            if entry.get("challenger_score"):
                score_parts.append(
                    f"Challenger: {entry['challenger_score'].title()}"
                )
            if entry.get("deal_trajectory"):
                score_parts.append(
                    f"Trajectory: {entry['deal_trajectory'].title()}"
                )
            if entry.get("talk_ratio") is not None:
                score_parts.append(
                    f"Talk Ratio: {entry['talk_ratio']:.0%}"
                )
            if score_parts:
                st.markdown("**Scores:** " + " | ".join(score_parts))

            tabs = st.tabs(["Analysis", "Transcript"])

            with tabs[0]:
                if entry.get("analysis"):
                    st.markdown(entry["analysis"])
                else:
                    st.info("No analysis stored.")

            with tabs[1]:
                if entry.get("diarized_transcript"):
                    st.markdown("**Speaker-Labeled Transcript:**")
                    st.markdown(entry["diarized_transcript"])
                elif entry.get("transcript"):
                    st.text_area(
                        "Transcript",
                        value=entry["transcript"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"hist_transcript_{entry['id']}",
                    )
                else:
                    st.info("No transcript stored.")

            # Delete button
            if st.button(
                "Delete this entry",
                key=f"del_{entry['id']}",
            ):
                delete_history_entry(entry["id"])
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Coaching Trends
# ---------------------------------------------------------------------------
def page_coaching_trends():
    """Display coaching score trends over time."""
    st.header("Coaching Trends")

    data = get_coaching_scores_over_time()

    if not data:
        st.info(
            "No coaching data yet. Process some **External Sales Call** "
            "recordings on the **Process** page to start tracking trends."
        )
        return

    st.caption(f"Tracking {len(data)} external sales calls over time")

    # Build DataFrame
    records = []
    for row in data:
        dt = row["created_at"][:10]
        records.append({
            "date": dt,
            "file": row["file_name"],
            "SPIN": SCORE_NUMERIC.get(row.get("spin_score", ""), None),
            "Challenger": SCORE_NUMERIC.get(
                row.get("challenger_score", ""), None
            ),
            "Talk Ratio": row.get("talk_ratio"),
            "spin_label": (row.get("spin_score") or "").title(),
            "challenger_label": (row.get("challenger_score") or "").title(),
            "trajectory": (row.get("deal_trajectory") or "").title(),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # --- SPIN & Challenger Scores Over Time ---
    st.subheader("SPIN & Challenger Scores Over Time")
    st.caption("Scale: 1 = Weak, 2 = Developing, 3 = Strong")

    score_df = df[["date", "SPIN", "Challenger"]].dropna(
        subset=["SPIN", "Challenger"], how="all"
    )

    if not score_df.empty:
        chart_data = score_df.set_index("date")[["SPIN", "Challenger"]]
        st.line_chart(chart_data, use_container_width=True)

        # Summary metrics
        cols = st.columns(4)
        with cols[0]:
            avg_spin = score_df["SPIN"].mean()
            st.metric(
                "Avg SPIN Score",
                f"{avg_spin:.1f}",
            )
        with cols[1]:
            avg_chall = score_df["Challenger"].mean()
            st.metric(
                "Avg Challenger Score",
                f"{avg_chall:.1f}",
            )
        with cols[2]:
            latest_spin = score_df["SPIN"].iloc[-1] if len(score_df) > 0 else 0
            first_spin = score_df["SPIN"].iloc[0] if len(score_df) > 0 else 0
            delta_spin = latest_spin - first_spin
            st.metric(
                "SPIN Trend",
                f"{latest_spin:.0f}",
                delta=f"{delta_spin:+.0f}" if delta_spin != 0 else "No change",
            )
        with cols[3]:
            latest_chall = (
                score_df["Challenger"].iloc[-1] if len(score_df) > 0 else 0
            )
            first_chall = (
                score_df["Challenger"].iloc[0] if len(score_df) > 0 else 0
            )
            delta_chall = latest_chall - first_chall
            st.metric(
                "Challenger Trend",
                f"{latest_chall:.0f}",
                delta=(
                    f"{delta_chall:+.0f}"
                    if delta_chall != 0
                    else "No change"
                ),
            )
    else:
        st.info("Not enough score data to chart yet.")

    # --- Talk Ratio Over Time ---
    st.divider()
    st.subheader("Talk Ratio Over Time")
    st.caption(
        "Percentage of conversation spoken by the rep. "
        "Ideal range: 40\u201360%."
    )

    talk_df = df[["date", "Talk Ratio", "file"]].dropna(subset=["Talk Ratio"])

    if not talk_df.empty:
        chart_talk = talk_df.set_index("date")[["Talk Ratio"]]
        st.line_chart(chart_talk, use_container_width=True)

        cols = st.columns(3)
        with cols[0]:
            avg_talk = talk_df["Talk Ratio"].mean()
            st.metric("Average Talk Ratio", f"{avg_talk:.0%}")
        with cols[1]:
            min_talk = talk_df["Talk Ratio"].min()
            st.metric("Lowest (Best Listening)", f"{min_talk:.0%}")
        with cols[2]:
            max_talk = talk_df["Talk Ratio"].max()
            st.metric("Highest (Most Talking)", f"{max_talk:.0%}")
    else:
        st.info(
            "No talk ratio data yet. Enable **Speaker Diarization** on the "
            "Process page to start tracking."
        )

    # --- Deal Trajectory Distribution ---
    st.divider()
    st.subheader("Deal Trajectory Distribution")

    traj_data = [
        r["trajectory"] for r in records if r.get("trajectory")
    ]
    if traj_data:
        traj_counts = pd.Series(traj_data).value_counts()
        st.bar_chart(traj_counts)
    else:
        st.info("No deal trajectory data available.")

    # --- Recent Calls Table ---
    st.divider()
    st.subheader("Recent Call Scores")

    table_df = df[
        [
            "date",
            "file",
            "spin_label",
            "challenger_label",
            "trajectory",
            "Talk Ratio",
        ]
    ].copy()
    table_df.columns = [
        "Date",
        "File",
        "SPIN",
        "Challenger",
        "Trajectory",
        "Talk Ratio",
    ]
    table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
    table_df["Talk Ratio"] = table_df["Talk Ratio"].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "\u2014"
    )
    table_df = table_df.sort_values("Date", ascending=False).head(20)
    st.dataframe(table_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Search
# ---------------------------------------------------------------------------
def _highlight_matches(text, query, context_chars=120):
    """Return a list of text snippets with the query highlighted in markdown bold."""
    if not text or not query:
        return []

    snippets = []
    lower_text = text.lower()
    lower_query = query.lower()
    start = 0

    while True:
        idx = lower_text.find(lower_query, start)
        if idx == -1:
            break

        # Extract surrounding context
        snippet_start = max(0, idx - context_chars)
        snippet_end = min(len(text), idx + len(query) + context_chars)

        before = text[snippet_start:idx]
        match = text[idx : idx + len(query)]
        after = text[idx + len(query) : snippet_end]

        prefix = "\u2026" if snippet_start > 0 else ""
        suffix = "\u2026" if snippet_end < len(text) else ""

        snippets.append(f"{prefix}{before}**{match}**{after}{suffix}")

        start = idx + len(query)

        if len(snippets) >= 3:
            break

    return snippets


def page_search():
    """Full-text search across all stored transcripts and analyses."""
    st.header("Search History")

    # Search controls
    col_query, col_mode = st.columns([3, 1])

    with col_query:
        query = st.text_input(
            "Search transcripts and analyses",
            placeholder="e.g., Carlisle, warehouse project, Pro-Grade 988...",
            key="search_query",
        )

    with col_mode:
        mode_filter = st.selectbox(
            "Filter by mode",
            options=["All", "External Sales Call", "Internal Meeting"],
            key="search_mode_filter",
        )

    if not query:
        st.info(
            "Enter a search term to find it across all stored transcripts, "
            "analyses, and file names."
        )
        return

    if len(query) < 2:
        st.warning("Please enter at least 2 characters.")
        return

    # Execute search
    results = search_history(query, mode_filter if mode_filter != "All" else None)

    if not results:
        st.warning(f'No results found for "{query}".')
        return

    st.caption(f'{len(results)} result{"s" if len(results) != 1 else ""} for "{query}"')

    for entry in results:
        created = entry["created_at"][:16].replace("T", " ")
        mode_tag = entry["mode"]

        # Determine where the match was found
        match_locations = []
        if entry.get("file_name") and query.lower() in entry["file_name"].lower():
            match_locations.append("filename")
        if entry.get("transcript") and query.lower() in entry["transcript"].lower():
            match_locations.append("transcript")
        if entry.get("analysis") and query.lower() in entry["analysis"].lower():
            match_locations.append("analysis")

        match_tag = ", ".join(match_locations) if match_locations else "match"
        label = (
            f"{entry['file_name']} \u2014 {mode_tag} \u2014 "
            f"{created} [{match_tag}]"
        )

        with st.expander(label, expanded=False):
            # Score summary
            score_parts = []
            if entry.get("spin_score"):
                score_parts.append(f"SPIN: {entry['spin_score'].title()}")
            if entry.get("challenger_score"):
                score_parts.append(
                    f"Challenger: {entry['challenger_score'].title()}"
                )
            if entry.get("deal_trajectory"):
                score_parts.append(
                    f"Trajectory: {entry['deal_trajectory'].title()}"
                )
            if entry.get("talk_ratio") is not None:
                score_parts.append(f"Talk Ratio: {entry['talk_ratio']:.0%}")
            if score_parts:
                st.markdown("**Scores:** " + " | ".join(score_parts))

            # Show highlighted snippets
            st.markdown("---")
            shown_snippet = False

            # Transcript matches
            transcript_text = (
                entry.get("diarized_transcript") or entry.get("transcript") or ""
            )
            transcript_snippets = _highlight_matches(transcript_text, query)
            if transcript_snippets:
                st.markdown("**Matches in transcript:**")
                for snippet in transcript_snippets:
                    st.markdown(f"> {snippet}")
                shown_snippet = True

            # Analysis matches
            analysis_snippets = _highlight_matches(
                entry.get("analysis", ""), query
            )
            if analysis_snippets:
                if shown_snippet:
                    st.markdown("")
                st.markdown("**Matches in analysis:**")
                for snippet in analysis_snippets:
                    st.markdown(f"> {snippet}")
                shown_snippet = True

            if not shown_snippet:
                st.markdown(f"*Match found in file name: {entry['file_name']}*")

            # Full content tabs
            st.markdown("---")
            tabs = st.tabs(["Full Analysis", "Full Transcript"])

            with tabs[0]:
                if entry.get("analysis"):
                    st.markdown(entry["analysis"])
                else:
                    st.info("No analysis stored.")

            with tabs[1]:
                if entry.get("diarized_transcript"):
                    st.markdown("**Speaker-Labeled Transcript:**")
                    st.markdown(entry["diarized_transcript"])
                elif entry.get("transcript"):
                    st.text_area(
                        "Transcript",
                        value=entry["transcript"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"search_transcript_{entry['id']}",
                    )
                else:
                    st.info("No transcript stored.")


# ---------------------------------------------------------------------------
# Streamlit App — Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="SalesBuddy.ai",
        page_icon="\U0001F3AF",
        layout="wide",
    )

    # Initialize database on startup
    init_database()

    # --- API key -----------------------------------------------------------
    api_key = get_api_key()

    # --- Page navigation (sidebar top) -------------------------------------
    with st.sidebar:
        page = st.radio(
            "Navigate",
            options=["Process", "History", "Search", "Coaching Trends"],
            horizontal=True,
        )

    # --- Page routing ------------------------------------------------------
    if page == "Process":
        st.title("SalesBuddy.ai")
        st.caption("Sales & Meeting Intelligence for Henry Company")

        if not api_key:
            st.error(
                "**OpenAI API key not found.** "
                "Please configure it in one of the following ways:"
            )
            st.info(
                "**Streamlit Community Cloud:** Add `OPENAI_API_KEY` in your "
                "app's *Settings > Secrets* panel.\n\n"
                "**Local development:** Create `.streamlit/secrets.toml` with:\n"
                "```\nOPENAI_API_KEY = \"sk-...\"\n```\n"
                "Or set the `OPENAI_API_KEY` environment variable."
            )
            st.stop()

        client = OpenAI(api_key=api_key)
        page_process(client)

    elif page == "History":
        st.title("SalesBuddy.ai")
        st.caption("Session History")
        page_history()

    elif page == "Search":
        st.title("SalesBuddy.ai")
        st.caption("Search Across All Sessions")
        page_search()

    elif page == "Coaching Trends":
        st.title("SalesBuddy.ai")
        st.caption("Coaching Performance Over Time")
        page_coaching_trends()


if __name__ == "__main__":
    main()
