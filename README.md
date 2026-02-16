# SalesBuddy.ai

Sales & Meeting Intelligence for Henry Company. Upload audio recordings from your Plaud Note device and get Salesforce-ready call logs, sales coaching insights, and structured meeting minutes — powered by OpenAI.

## Features

- **Batch upload** — process multiple WAV, MP3, or M4A recordings at once
- **Two analysis modes:**
  - **External Sales Call** — generates a Salesforce activity log (subject, account, relationship status, products discussed, next steps) plus SPIN Selling and Challenger Sale coaching analysis
  - **Internal Meeting** — generates meeting minutes, action items with owners, and decisions made
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
3. Select the **Processing Mode**:
   - *External Sales Call* — for contractor, distributor, or owner calls
   - *Internal Meeting* — for pipeline reviews, strategy sessions, etc.
4. Click **Process Files**.
5. Review results in the expandable sections. Each file has an **Analysis** tab and a **Transcript** tab.
6. Copy the Salesforce block into your CRM, review coaching feedback, and share meeting minutes with your team.

---

## Cost Estimates

| Operation     | Model     | Approximate Cost                          |
|---------------|-----------|-------------------------------------------|
| Transcription | whisper-1 | $0.006 per minute of audio                |
| Analysis      | gpt-4o    | ~$0.01–0.03 per call (varies by length)   |

**Example:** A 30-minute sales call costs roughly **$0.20** ($0.18 transcription + ~$0.02 analysis).

---

## Project Structure

```
salesbuddy.ai/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes secrets and artifacts
├── .streamlit/
│   └── config.toml        # Theme and upload size settings
└── README.md              # This file
```
