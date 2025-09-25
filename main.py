from pathlib import Path

import click

from voxmem.input import LocalInput
from voxmem.output import LocalJsonOutput
from voxmem.transcription import NoopTranscriber


@click.command()
@click.argument("input", type=click.Path())
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="transcripts",
    show_default=True,
    help="Directory to save transcription results",
)
def main(input, out_dir):
    inp = LocalInput(Path(input))
    out = LocalJsonOutput(root=Path(out_dir))
    transcriber = NoopTranscriber()

    produced = False
    for source in inp:
        produced = True

        name = Path(source).stem
        if out.exists(name):
            click.echo(f"skip: {out.path_for(name)}")
            continue

        result = transcriber.transcribe(source, out_dir=out_dir)
        saved = out.save(name, {"result": result})
        click.echo(saved)

    if not produced:
        raise click.UsageError("No audio files found for the given input.")


if __name__ == "__main__":
    main()
