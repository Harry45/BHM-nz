"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: Function to process the data in the right format.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""

import sys
import pandas as pd

from configuration import config

CONFIG = config()


def get_size_mb(dataframe: pd.DataFrame):
    """Print the size of the pandas dataframe in MB

    Args:
        dataframe (pd.DataFrame): the pandas dataframe
    """
    print(f'Size of dataframe {sys.getsizeof(dataframe)/1024**2:.2f} MB')


def split_data(cataloguengalaxies: int, save: bool) -> dict:
    """Split the full data into sets of galaxies. These are randomly chosen from the full catalogue.

    Args:
        ngalaxies (int): _description_
        save (bool): _description_

    Returns:
        dict: _description_
    """
