"""Example of converting I3-files to SQLite and Parquet."""

import os
from glob import glob

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors.icecube import (
    I3GenericExtractor,
    I3FeatureExtractor
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

ERROR_MESSAGE_MISSING_ICETRAY = (
    "This example requires IceTray to be installed, which doesn't seem to be "
    "the case. Please install IceTray; run this example in the GraphNeT "
    "Docker container which comes with IceTray installed; or run an example "
    "script in one of the other folders:"
    "\n * examples/02_data/"
    "\n * examples/03_weights/"
    "\n * examples/04_training/"
    "\n * examples/05_pisa/"
    "\nExiting."
)

CONVERTER_CLASS = {
    "sqlite": SQLiteDataConverter,
    "parquet": ParquetDataConverter,
}


def main_pone(backend: str) -> None:
    """Convert p-one I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs = ["/home/victoria/work/pone/samples"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone"
    print("outdir: ", outdir)
    gcd_rescue = glob(
        "/home/victoria/work/pone/samples/*GCD*"
    )
    if not gcd_rescue:
        print("No GeoCalib files found for p-one.")
        return
    gcd_rescue = gcd_rescue[0]
    print("gcd type", type(gcd_rescue))
    converter = CONVERTER_CLASS[backend](
        extractors=[
            I3FeatureExtractor(),
            I3GenericExtractor(),

        ],
        outdir=outdir,
        gcd_rescue=gcd_rescue,
        workers=1,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files()

if __name__ == "__main__":

    if not has_icecube_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_ICETRAY)
    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
Convert I3 files to an intermediate format.
"""
        )

        parser.add_argument("backend", choices=["sqlite", "parquet"])
        # parser.add_argument("outdir", default=f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone")
        args, unknown = parser.parse_known_args()
        

        # Run example script
        main_pone(args.backend)

