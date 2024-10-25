from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR

from graphnet.utilities.logging import Logger
from icecube import dataclasses, dataio

import pandas as pd

def convert_gcd_to_csv(gcd_file, csv_file):
    # Load the GCD file
    gcd_reader = dataio.I3File(gcd_file)
    gcd_frame = gcd_reader.pop_frame()

    # Define the variables
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"
    pmt_number_column = "pmt_number"
    dom_type_column = "dom_type"

    # Extract the necessary information from the GCD frame and write it to the CSV file
    with open(csv_file, "w") as f:
        # Write the header row
        f.write(f"{string_id_column},{sensor_id_column},{pmt_number_column},{xyz[0]},{xyz[1]},{xyz[2]},{dom_type_column}\n")
        for omkey, dom_geom in gcd_frame["I3Geometry"].omgeo.items():
            dom_pos = dom_geom.position
            string_id = omkey.string
            sensor_id = omkey.om
            pmt_number = omkey.pmt
            dom_type = dom_geom.omtype
            f.write(f"{string_id},{sensor_id},{pmt_number},{dom_pos.x},{dom_pos.y},{dom_pos.z},{dom_type}\n")
    Logger().info(f"Converted GCD file to CSV: {csv_file}")

    return xyz

def conver_to_parquet(csv_file, parquet_file, xyz):

    table_without_index = pd.read_csv(csv_file)
    geometry_table = table_without_index.set_index(xyz)
    geometry_table.to_parquet(parquet_file)
    Logger().info(f"Converted CSV file to Parquet: {parquet_file}")

def check_parquet(parquet_file):
    geometry_table = pd.read_parquet(parquet_file)
    print(geometry_table)

if __name__ == "__main__":
    gcd_file = f"{TEST_DATA_DIR}/i3/pone-GenerateSingleMuons_39_10String_7Cluster/PONE_10String_7Cluster_standard_GCD.i3.gz"
    csv_file = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone/pone.csv"  # Specify the path to the output CSV file
    xyz = convert_gcd_to_csv(gcd_file, csv_file)
    parquet_file = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone/pone.parquet"  # Specify the path to the output Parquet file
    conver_to_parquet(csv_file, parquet_file, xyz)
    check_parquet(parquet_file)