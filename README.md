# Transcriptions

A simple transcription pipeline powered by [ElevenLabs Scribe v2](https://elevenlabs.io/docs/overview/models#scribe-v2) or [AssemblyAI](https://www.assemblyai.com/).

The command-line application reads a CSV file containing `filename` and `url` columns, submits every audio URL to AssemblyAI, persists the full JSON response for each row, and stores the returned transcription ID back into the source CSV.

## Installation

```bash
uv sync
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Configuration

Add your transcription provider token to the environment:

```
ELEVENLABS_API_KEY=sk-...
ASSEMBLYAI_API_KEY=sk-...
```

The CLI automatically loads `.env` files via `python-dotenv`.

## CSV format

The input CSV must contain two columns:

```csv
filename,url
01_audio.mp3,https://example.com/01_audio.mp3
```

During execution the tool adds/updates the following columns atomically:

- `transcription_id` – provider transcript ID
- `status` – `completed` or `error`
- `error` – most recent error message (if any)

Duplicate filenames (within the selected `offset`/`limit` window) cause the run to abort to avoid clobbering outputs.

## Usage

Run the Typer-based CLI with `uv` (recommended). ElevenLabs is the default provider and uses batch-style polling if the API responds before word timestamps are ready. This tool stores only the JSON transcription payloads.

```bash
uv run python main.py transcribe tmp/samples/samples.csv \
  --output output/runs/batch-001 \
  --workers 25 \
  --offset 0 \
  --limit 250 \
  --logfile output/runs/batch-001/transcriptions.log
```

To use AssemblyAI instead:

```bash
uv run python main.py transcribe tmp/samples/samples.csv \
  --provider assemblyai \
  --output output/runs/batch-001 \
  --workers 25
```

Use `uv run python main.py --help` for the complete list of options.

To only validate the CSV slice for duplicate filenames without transcribing, run:

```bash
uv run python main.py check tmp/samples/samples.csv
```

To export completed rows to a minimal CSV with download URLs:

```bash
uv run python main.py export tmp/samples/samples.csv --output tmp/export.csv
```

To report total size and estimated duration for URLs in the CSV file:

```bash
uv run python main.py stats tmp/samples/samples.csv --workers 50
```
