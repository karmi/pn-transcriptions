# Transcriptions

Transcribe audio files to text. Currently supports files stored locally or in the Cloudflare R2 storage, and uses the [AssemblyAI](https://www.assemblyai.com/) service for transcriptions.

## Installation

Requires Python 3.13+, [`uv`](https://docs.astral.sh/uv/) for managing dependencies and an AssemblyAI account (freemium available).

Create a virtual environment and install the package:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Configuration

Export the following environment variables:

- `ASSEMBLYAI_API_KEY` — AssemblyAI access token
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` — Cloudflare R2 keys for remote inputs

## Running Locally

The CLI consumes a single audio file, a directory tree, or an `r2://bucket/prefix` URL and writes JSON transcripts to the target directory (default `transcripts/`). Existing outputs are skipped automatically.

```bash
uv run python main.py <input>
```

Use `uv run main.py --help` for the list of options.
