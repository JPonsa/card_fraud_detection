"""
This scripts download simulated card frau detection data from github repo
and converts it into a single parquet file in the raw layer
"""

import glob
import os
import pandas as pd
import requests
import shutil
import zipfile


def download_from_url(url: str, file_name: str, destination_dir: str) -> None:
    """donwload file from url

    Parameters
    ----------
    url : str
    file_name : str
    destination_dir : str
        path to where to save the file(s)
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Download the file and save it in data_path
    response = requests.get(url)
    with open(os.path.join(url, file_name), "wb") as file:
        file.write(response.content)


def pickles_2_parquet(source_dir: str, destination_dir: str, parquet_file: str) -> None:
    """Takes a directory with multiple pickle files and merge them in a single parquet file.

    Parameters
    ----------
    source_dir : str
        path to where the picket files are
    destination_dir : str
        path to where the parquet file needs to be saved
    parquet_file : str
        name of the parquet file. It must end in extension ".parquet.gzip"
    """
    pkl_files = glob.glob(os.path.join(source_dir, "*.pkl"))

    # Create the destination_dir directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Loop through all .pkl files and append them to the DataFrame
    for file in pkl_files:
        df_temp = pd.read_pickle(file)
        df = pd.concat([df, df_temp], ignore_index=True)

    df.to_parquet(destination_dir + parquet_file, compression="gzip")


if __name__ == "__main__":
    raw_path = "./data/raw/"
    url = "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/archive/main.zip"

    download_from_url(url, "main.zip", "./tmp/")

    # unzip data
    zip_path = "./tmp/main.zip"

    # Create a ZipFile Object
    with zipfile.ZipFile(zip_path) as zip_ref:
        # Extract all the contents of zip file in current directory
        zip_ref.extractall("./tmp/")

    # Get a list of all .pkl files in the data_path
    source_dir = "./tmp/simulated-data-raw-main/data/"
    pickles_2_parquet(source_dir, raw_path, "card_fraud.parquet.gzip")

    # clear tmp directory
    shutil.rmtree("./tmp/")
