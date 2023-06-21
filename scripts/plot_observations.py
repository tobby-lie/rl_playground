"""Utility script used for plotting and comparing output elements of observations"""
import json
import logging
from collections.abc import Iterator, Sequence
from functools import partial
from pathlib import Path

import click
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.util import validate_click_file_suffix

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--results-file",
    "-l",
    help=".json file path containing list of results with observations to plot",
    type=Path,
    required=True,
    callback=partial(validate_click_file_suffix, suffix=".json"),
)
@click.option(
    "--output-dir",
    "-o",
    help="Output directory for plot artifact files",
    type=Path,
    required=True,
)
@click.option(
    "--artifacts-limit",
    "-a",
    help="Limit for number of artifacts to generate",
    type=int,
    required=True,
)
def main(results_file: Path, output_dir: Path, artifacts_limit: int):
    """TODO:"""
    logger.info(f"Creating {output_dir=}")
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Hydrating {results_file=}")
    with open(str(results_file)) as rf:
        result_observations: Iterator[Sequence[Sequence[float]]] = (
            result["observations"] for result in json.load(rf)
        )
    # TODO: key each of the outputs for each observation ->
    #   [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    logger.info(f"Generating {artifacts_limit=} plots")
    for i, observations in tqdm(
        enumerate(result_observations, start=1), total=artifacts_limit
    ):
        if i > artifacts_limit:
            break
        # TODO: This could benefit form keys instead of numeric indices
        plt.clf()
        plt.plot(
            tuple(
                observation[0] for observation in observations
            ),
            color="r",
            label="cart position"
        )
        plt.plot(
            tuple(
                observation[1] for observation in observations
            ),
            color="g",
            label="cart velocity"
        )
        plt.legend(loc="upper left")
        plt.savefig(str(Path(output_dir) / f"trial_{i}.png"))
    logger.info(
        f"Stopped generating plots at trial {i}\n"
        f"Total number of observations: {i}\n"
        f"Artifacts limit: {artifacts_limit}"
    )


if __name__ == "__main__":
    main()
