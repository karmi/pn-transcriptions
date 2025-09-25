import click

from voxmem.storage import LocalSource
from voxmem.transcription import NoopTranscriber


@click.command()
@click.argument("path", type=click.Path())
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="transcripts",
    show_default=True,
    help="Directory to save transcription results",
)
def main(path, out_dir):
    """Generate text transcriptions for audio files"""
    source = LocalSource().resolve(path)
    transcriber = NoopTranscriber()
    result = transcriber.transcribe(source, out_dir=out_dir)
    click.echo(result)


if __name__ == "__main__":
    main()
