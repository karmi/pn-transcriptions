# Transcriptions

A simple transcription pipeline powered by [AssemblyAI](https://www.assemblyai.com/).

The command-line application reads a CSV file containing `filename` and `url` columns, submits every audio URL to AssemblyAI, persists the full JSON response for each row, and stores the returned transcription ID back into the source CSV.

## Installation

```bash
uv sync
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Configuration

Add your AssemblyAI token to the environment:

```
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

- `transcription_id` – AssemblyAI transcript ID
- `status` – `completed` or `error`
- `error` – most recent error message (if any)

Duplicate filenames (within the selected `offset`/`limit` window) cause the run to abort to avoid clobbering outputs.

## Usage

Run the Typer-based CLI with `uv` (recommended):

```bash
uv run python main.py tmp/samples/samples.csv \
  --output output/runs/batch-001 \
  --workers 8 \
  --offset 0 \
  --limit 250 \
  --logfile output/runs/batch-001/transcriptions.log
```

Use `uv run python main.py --help` for the complete list of options.
