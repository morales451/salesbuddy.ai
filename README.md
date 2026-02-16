# SalesBuddy.ai

Sales & Meeting Intelligence for Henry Company. Upload audio recordings from your Plaud Note device and get Salesforce-ready call logs, sales coaching insights, competitive intelligence, and structured meeting minutes — powered by OpenAI.

## Features

- **Batch upload** — process multiple WAV, MP3, or M4A recordings at once
- **Per-file mode selection** — assign External Sales Call or Internal Meeting mode to each file independently
- **Two analysis modes:**
  - **External Sales Call** — Salesforce activity log, SPIN/Challenger coaching, competitive intelligence, and objection handling analysis
  - **Internal Meeting** — meeting minutes, action items with owners, and decisions made
- **Visual coaching scorecard** — at-a-glance SPIN Selling, Challenger Sale, and Deal Trajectory gauges for each external call
- **Competitive intelligence** — automatic extraction of competitor mentions, switching signals, and market intel
- **Objection handling analysis** — identifies objections, evaluates rep responses, and suggests stronger alternatives
- **Copy-to-clipboard** — one-click copy for Salesforce logs, competitive intel, objection analysis, and more
- **Download & export** — download individual analyses, transcripts, or a combined full report as markdown
- **CSV export** — export Salesforce fields (Subject, Account, Status, Products, Next Steps) as a CSV for bulk import
- **End-of-day pipeline summary** — generate a roll-up briefing across all processed calls: accounts touched, aggregate next steps, pipeline health, and top priorities for tomorrow
- **Follow-up email drafts** — auto-generate a professional follow-up email based on any external call's analysis
- **Salesforce API integration** — push call logs directly to Salesforce as completed Task records (optional, requires credentials)
- **Auto-chunking** — files over 25 MB are automatically split so they fit within the Whisper API limit
- **Henry Company context** — the AI is primed with your product portfolio, industry terms, and territory details

---

## Local Development Setup

### Prerequisites

- Python 3.9+
- `ffmpeg` (required by `pydub` for audio processing)
  - **macOS:** `brew install ffmpeg`
  - **Ubuntu / Debian:** `sudo apt-get install ffmpeg`
  - **Windows:** download from <https://ffmpeg.org/download.html> and add to PATH

### Installation

```bash
git clone <your-repo-url>
cd salesbuddy.ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure your API key

Create `.streamlit/secrets.toml` (this file is git-ignored):

```toml
OPENAI_API_KEY = "sk-..."
```

Or set the environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud

1. **Push this repository** to a GitHub repo (the `.gitignore` already excludes secrets).

2. Go to **<https://share.streamlit.io>** and click **New app**.

3. Connect your GitHub repo and set:
   - **Main file path:** `app.py`

4. In **Advanced settings > Secrets**, add:

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

5. Click **Deploy**. The app will install dependencies from `requirements.txt` automatically. `ffmpeg` is pre-installed on Streamlit Cloud's runtime.

---

## Usage

1. Open the app in your browser.
2. In the **sidebar**, upload one or more audio files (WAV, MP3, or M4A).
3. Select the **Default Processing Mode** — or expand "Per-file mode overrides" to set each file individually.
4. Click **Process Files**.
5. Review results:
   - **Coaching Scorecard** — visual gauges for SPIN, Challenger, and Deal Trajectory (external calls)
   - **Analysis tab** — full analysis with download and copy buttons
   - **Transcript tab** — raw transcript with download
   - **Salesforce Log tab** — isolated Salesforce block with one-click copy
   - **Competitive Intel tab** — competitor mentions and market intelligence
   - **Objections tab** — objection handling analysis with coaching suggestions
   - **Follow-Up Email tab** — generate and copy a professional follow-up email
   - **Push to SF tab** — push the call log directly to Salesforce
6. Use the **global action bar** to download the full report, export Salesforce CSV, or generate a pipeline summary.

---

## Salesforce Integration (Optional)

To push call logs directly to Salesforce:

1. Expand **Salesforce Integration** in the sidebar.
2. Enter your Salesforce username, password, and security token.
3. Select the domain (`login` for production, `test` for sandbox).
4. Use the **Push to SF** tab on any external call result.

The integration creates a completed Task record in Salesforce with the call subject and full analysis in the description.

---

## Cost Estimates

| Operation         | Model     | Approximate Cost                          |
|-------------------|-----------|-------------------------------------------|
| Transcription     | whisper-1 | $0.006 per minute of audio                |
| Analysis          | gpt-4o    | ~$0.01–0.03 per call (varies by length)   |
| Pipeline Summary  | gpt-4o    | ~$0.02–0.05 per summary                   |
| Follow-Up Email   | gpt-4o    | ~$0.005 per email                         |

**Example:** A 30-minute sales call costs roughly **$0.20** ($0.18 transcription + ~$0.02 analysis).

---

## Project Structure

```
salesbuddy.ai/
├── app.py                 # Streamlit application (all features)
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes secrets and artifacts
├── .streamlit/
│   └── config.toml        # Theme and upload size settings
└── README.md              # This file
```
