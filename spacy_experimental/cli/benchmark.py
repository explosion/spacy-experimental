from pathlib import Path
from typing import Iterable
from spacy import Language, util
from spacy.cli import app
from spacy.tokens import Doc
from spacy.training import Corpus
from thinc.api import require_gpu
from thinc.util import gpu_is_available
import time
from typer import Argument as Arg, Option
from wasabi import Printer


@app.command("benchmark")
def benchmark_cli(
    model: str = Arg(..., help="Model name or path"),
    data_path: Path = Arg(
        ..., help="Location of binary evaluation data in .spacy format", exists=True
    ),
    use_gpu: int = Option(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
    warmup_epochs: int = Option(
        3, "--warmup", "-w", min=0, help="Number of iterations over the data for warmup"
    ),
    bench_epochs: int = Option(
        3,
        "--iter",
        "-i",
        min=1,
        help="Number of iterations over the data for benchmarking",
    ),
):
    """
    Benchmark a pipeline. Expects a loadable spaCy pipeline and benchmark
    data in the binary .spacy format.
    """
    setup_gpu(use_gpu=use_gpu, silent=False)

    nlp = util.load_model(model)
    corpus = Corpus(data_path)
    docs = [eg.predicted for eg in corpus(nlp)]
    n_tokens = count_tokens(docs)

    warmup(nlp, docs, warmup_epochs)

    best = benchmark(nlp, docs, bench_epochs)

    print(
        "Best of %d runs: %.3fs %.0f, words/s" % (bench_epochs, best, n_tokens / best)
    )


def annotate(nlp: Language, docs: Iterable[Doc]):
    for _ in nlp.pipe(docs, batch_size=256):
        pass


def benchmark(nlp: Language, docs: Iterable[Doc], bench_epochs: int):
    best = float("inf")
    for _ in range(bench_epochs):
        start = time.time()
        annotate(nlp, docs)
        end = time.time()
        run = end - start
        if run < best:
            best = run
    return best


def count_tokens(docs: Iterable[Doc]):
    count = 0
    for doc in docs:
        for _ in doc:
            count += 1
    return count


def warmup(nlp: Language, docs: Iterable[Doc], warmup_epochs: int):
    for _ in range(warmup_epochs):
        annotate(nlp, docs)


def setup_gpu(use_gpu: int, silent=None) -> None:
    """Configure the GPU and log info."""
    if silent is None:
        local_msg = Printer()
    else:
        local_msg = Printer(no_print=silent, pretty=not silent)
    if use_gpu >= 0:
        local_msg.info(f"Using GPU: {use_gpu}")
        require_gpu(use_gpu)
    else:
        local_msg.info("Using CPU")
        if gpu_is_available():
            local_msg.info("To switch to GPU 0, use the option: --gpu-id 0")
