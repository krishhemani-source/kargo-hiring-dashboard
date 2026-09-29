# Kargo Hiring Dashboard

Screens CVs for **Product Manager (PM)** and **Senior Product Manager (SPM)** against Kargo's rubric. The founder then sends an interview invite or a rejection. The AI scores and drafts. Arjun decides. Nothing is sent until he clicks **Send**.

| Actor | Step | Where in the code |
|---|---|---|
| Founder | **Trigger / Input**: upload one or many PDF/DOCX files and pick PM / SPM | `static/index.html`, `POST /api/upload` |
| Backend | **Context**: extract text; pull name/email/phone/links/city into their own columns; **replace them with `[NAME]`, `[EMAIL]`, `[PHONE]`, `[LINK]`, `[CITY]`** | `app/parsing.py` |
| Backend | **Processing**: score against **both** rubrics (one LLM call per rubric, strict JSON, validated, retried once). The weighted % is calculated by the backend. The relocation gate is a flag and is not scored | `app/pipeline.py`, `app/rubric.py`, `app/llm.py` |
| LLM (Gemini) | **AI**: interview brief (who, why ranked here, 3 probes on weakest criteria) plus invite and rejection drafts written with `[NAME]`. The backend fills in the first name | `app/llm.py` |
| Resend | **Send**: only from the Send button. `TEST_MODE` redirects every email to `FOUNDER_EMAIL` | `app/emailer.py`, `POST /api/candidates/{id}/send` |
| Founder | **Output**: PM / SPM tabs ranked by weighted %, evidence per criterion, brief, editable drafts, audit trail, "haven't heard back" counter | `static/index.html` |

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then fill in `.env`:

- `GEMINI_API_KEY`: from https://aistudio.google.com/apikey
- `RESEND_API_KEY`: from resend.com/api-keys
- `FOUNDER_EMAIL`: your email. On Resend's free tier this **must be the email your Resend account is registered with**, because `onboarding@resend.dev` only delivers to that address. To email real candidates, verify a domain in Resend, set `FROM_EMAIL=Arjun <arjun@yourdomain.com>` and set `TEST_MODE=false`.
- `TEST_MODE=true` (the default) sends every email to `FOUNDER_EMAIL` with `[TEST → candidate@email]` at the start of the subject.

## Run

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

Open http://localhost:8000. To try it quickly, drag the files from `applicants/` into the upload box (pick PM or SPM first).

## Deploy (GitHub + Vercel + Neon)

On Vercel the app keeps its data in **Neon Postgres** (`DATABASE_URL`), including the original CV files. Locally it uses `data/kargo.db` (SQLite) unless `DATABASE_URL` is set. Scoring runs one request per CV, and the open dashboard sends those requests itself, because Vercel stops background work once a request returns. So **keep the dashboard tab open while a batch is scoring.** Any CVs still queued when you close it resume the next time you open the dashboard.

1. **GitHub:** push this folder to a **private** repo. `seed/` and `applicants/` contain CVs, so don't make the repo public. `.env` is git-ignored.
2. **Vercel:** click **Add New → Project**, import the repo, and leave the framework on **FastAPI** (it reads `pyproject.toml`, and the entrypoint is `app.main:app`).
3. **Environment variables** (Project → Settings → Environment Variables). You can paste them as a block:
   - `GEMINI_API_KEY`, `RESEND_API_KEY`, `FOUNDER_EMAIL`, `TEST_MODE=true`
   - Optional: `DASHBOARD_PASSWORD` turns on a login screen. Without it, anyone with the URL can open the dashboard.
   - Optional: `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS`, `FROM_EMAIL`, `FOUNDER_NAME`, `SESSION_SECRET`
4. **Neon:** open Project → **Storage → Create Database → Neon (Serverless Postgres)** and connect it to the project. This adds `DATABASE_URL` automatically. Tables are created on first use.
5. **Redeploy** so the new variables take effect, then open the `.vercel.app` URL and log in.

`vercel.json` gives each request up to 300 s (`maxDuration`) and keeps `seed/`, `applicants/` and `tests/` out of the deployed bundle.

## Calibration test

```bash
.venv/bin/python calibrate.py
```

This scores the 8 past hires in `seed/` on both rubrics and prints a ranked table. It checks two things:

1. Every Exceeds hire (Rohan, Sunita, Aditya, Meghna, Lavanya) ranks above every Meets hire (Vikram, Rahul) and the Below hire (Preetham).
2. On the PM rubric, Lavanya is at least 20 points above Vikram.

The two PM hires are ranked on their PM score. The other six were hired into other functions (engineering, ops consulting, sales, CS, dev, marketing), so they're ranked on their best-fit score (the higher of PM and SPM). If a check fails, the script prints the out-of-order pairs and the criteria that caused it. Scores are cached in `data/calibration_cache.json` by file hash, so a rerun costs nothing. Use `--fresh` to re-score. The script exits non-zero on failure.

## Tests

```bash
.venv/bin/python -m pytest -q
```

- `tests/test_pii.py` redacts every CV in `seed/` and `applicants/` and asserts no email, phone, URL or name token is left. It also intercepts the real LLM call path and checks the prompts, and checks that unredacted text is refused before any API call.
- `tests/test_scoring.py` covers rubric parsing, the backend weighted-% maths, the JSON validation and single retry, the rejection-can't-mention-scores check, and the relocation gate.
- `tests/test_api.py` runs the full flow with the LLM and Resend stubbed: batch upload with bad files, scoring, the decision, edits, a TEST_MODE send, no double sends, and duplicate detection.

## Guardrails

- **No personal data reaches the LLM.** `llm._call()` runs `assert_no_pii()` on every prompt. It blocks any email, phone number or URL pattern, the candidate's full name, and any single name token in the CV part of the prompt. If redaction ever misses something, the candidate is marked `failed` with the reason and nothing is sent to the API. Location is personal data too, so the relocation gate runs in the backend from the extracted city and "willing to relocate" wording. Job locations further down the CV (e.g. "Mahindra Logistics, Mumbai") are kept because they're work history.
- **The LLM never decides.** It returns criterion scores with evidence, plus drafts. Move forward / Pass and Send are founder-only actions, and each is logged with a timestamp and optional note (`events` table).
- **The LLM doesn't do the maths.** Weighted % = Σ(score × weight) / 5, in `rubric.weighted_percent()`.
- **Rejections never mention scores.** Drafts that mention %, scores or the rubric are rejected and regenerated once.
- **Messy CVs.** Fake-bold PDFs and edited contact headers (text pasted over text) are handled by rebuilding lines from content-stream runs. If a file fails, only that file is marked `failed` (with a Re-score button) and the rest of the batch continues. Duplicate uploads are detected by file hash.
- **Keys** live only in `.env` (git-ignored).

## Statuses

`new` (processing) → `scored` → `invite_ready` / `reject_ready` → `sent`. A file that couldn't be parsed or scored is `failed`. The header counter shows how many candidates haven't heard back yet. The goal is zero.

## Data

- Database: Neon Postgres when `DATABASE_URL` is set, otherwise SQLite at `data/kargo.db`. Original CV files are stored in the database.
- Model: `GEMINI_MODEL` (default `gemini-3.7-flash`), then `GEMINI_FALLBACK_MODELS` in order if a model is overloaded or out of its daily quota. The free tier allows only about 20 requests per model per day, and each CV takes 3 requests, so enable billing on the key for real batches. The app uses Gemini's JSON response mode, then still validates the JSON and retries once.
