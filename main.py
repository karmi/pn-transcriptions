from pathlib import Path

import click
from concurrent.futures import ThreadPoolExecutor, as_completed

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
@click.option(
    "--workers",
    "-j",
    type=int,
    default=5,
    show_default=True,
    help="Number of parallel workers for transcriptions",
)
def main(input, out_dir, workers):
    inp = LocalInput(Path(input))

    def _process(source: str) -> str:
        name = Path(source).stem
        out = LocalJsonOutput(root=Path(out_dir))
        if out.exists(name):
            return f"skip: {out.path_for(name)}"
        transcriber = NoopTranscriber()
        result = transcriber.transcribe(source, out_dir=out_dir)
        saved = out.save(name, {"result": result})
        return saved

    produced = False
    submitted = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for source in inp:
            produced = True
            futures.append(ex.submit(_process, source))
            submitted += 1

        for fut in as_completed(futures):
            click.echo(fut.result())

    if produced and submitted == 0:
        click.echo("All outputs exist; nothing to do.")

    if not produced:
        raise click.UsageError("No audio files found for the given input.")


if __name__ == "__main__":
    main()
