import glob
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from utils import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_multiple_dataframe(
    input_folder_path,
    output_folder_path
):
    """
    Merge multiple csv files into one dataframe, remove duplicates and save to disk

    Args:
        input_folder_path (str): path to the input folder
        output_folder_path (str): path to the output folder to save the data
    """
    # check for datasets, compile them together, and write to an output file
    csv_files = glob.glob(os.path.join(input_folder_path, '*.csv'))
    merged_data = pd.concat([pd.read_csv(file_path)
                            for file_path in csv_files], ignore_index=True)
    ingested_data = merged_data.drop_duplicates()
    os.makedirs(output_folder_path, exist_ok=True)
    ingested_data.to_csv(
        os.path.join(
            output_folder_path,
            'finaldata.csv'),
        index=False)

    logger.info(f"[Ingestion] Merging {csv_files} to {output_folder_path}")
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as f:
        for file_path in csv_files:
            f.write(os.path.basename(file_path) + "\n")


if __name__ == '__main__':
    cfg = Config()
    merge_multiple_dataframe(cfg.input_folder_path, cfg.output_folder_path)
