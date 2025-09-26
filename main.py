from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from tqdm import tqdm

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

    out = LocalJsonOutput(root=Path(out_dir))
    sources = list(inp)

    if not sources:
        raise click.UsageError("No audio files found for the given input.")

    to_run: list[str] = []
    skipped = 0
    for src in sources:
        name = Path(src).stem
        if out.exists(name):
            skipped += 1
            tqdm.write(f"skip: {out.path_for(name)}")
        else:
            to_run.append(src)

    if not to_run:
        click.echo("All outputs exist; nothing to do.")
        return

    def _process(source: str) -> str:
        transcriber = NoopTranscriber()
        name = Path(source).stem
        result = transcriber.transcribe(source, out_dir=out_dir)
        saved = out.save(name, {"result": result})
        return saved

    done = 0
    errs = 0

    with (
        tqdm(
            total=len(to_run),
            desc="transcribing",
            unit="file",
            dynamic_ncols=True,
        ) as bar,
        ThreadPoolExecutor(max_workers=workers) as ex,
    ):
        futures = [ex.submit(_process, src) for src in to_run]

        for fut in as_completed(futures):
            try:
                saved = fut.result()
                done += 1
                tqdm.write(f"done: {saved}")
            except Exception as e:
                errs += 1
                tqdm.write(f"error: {e!r}")
            finally:
                bar.set_postfix({"done": done, "skip": skipped, "err": errs})
                bar.update(1)


if __name__ == "__main__":
    main()
