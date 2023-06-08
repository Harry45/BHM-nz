"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: Function to process the data in the right format.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""

import sys
import pandas as pd
import numpy as np
from astropy.io import fits
from hurry.filesize import size
from ml_collections.config_dict import ConfigDict

# our scripts and functions
import utils.helpers as hp


def get_size_mb(dataframe: pd.DataFrame, name: str):
    """Print the size of the pandas dataframe in MB

    Args:
        dataframe (pd.DataFrame): the pandas dataframe
        name (str): name of the dataframe
    """
    filesize = size(sys.getsizeof(dataframe))
    print(f"Size of {name} dataframe {filesize}")


def extract_data(config: ConfigDict) -> dict:
    """
    Extract the data from the fits file. We also keep only the data points for which the
    magnitude values are below the magnitude limits.

    Args:
        config (ConfigDict): the main configuration file

    Returns:
        dict: a dictionary containing all the important quantities
    """
    fits_image = fits.open(config.paths.fitsfile)
    data = fits_image[1].data
    fits_image.close()

    quantities = dict(config.colnames)
    record = dict()

    # extract all the data points
    for qname in quantities:
        columns = quantities[qname]
        data_extracted = np.asarray([data[columns[i]] for i in range(len(columns))]).T
        record[qname] = pd.DataFrame(
            data_extracted, columns=columns, dtype=config.dtypes[qname]
        )

    # choose rows for which the magnitudes are within the magnitude limit
    condition = np.sum((record["mag"].values < record["maglim"].values) * 1, axis=1)
    condition = (condition == 9) * 1

    # apply the cuts
    for qname in quantities:
        record[qname] = record[qname][condition == 1].reset_index(drop=True)
    return record


def correct_data(config: ConfigDict, data: dict) -> dict:
    """
    Correct the data by shifting the fluxes according to the median of each patch/tile.

    Args:
        config (ConfigDict): the main configuration file

        data (dict): the processed data with keys:
         - 'extinction',
         - 'flux',
         - 'fluxerr',
         - 'mag',
         - 'magerr',
         - 'maglim',
         - 'redshift',
         - 'theliname'

         See the main configuration file. They are already defined there.

    Returns:
        dict: a dictionary containing the same keys and corresponding values but the with the correct data.
    """
    unique_names = np.unique(data["theliname"])
    nunique = len(unique_names)
    print(f"Number of tiles is: {nunique}")

    assert (
        config.ntiles < nunique
    ), "The number of tiles is greater than the number of available tiles."
    tiles = dict()
    for i in range(config.ntiles):
        record_tile = dict()
        tile = data["theliname"] == unique_names[i]
        tile = tile.values

        record_tile["mag"] = data["mag"][tile] - data["extinction"][tile].values
        record_tile["magerr"] = data["magerr"][tile]

        scaled_magnitude = data["mag"][tile] + 2.5 * np.log10(data["flux"][tile].values)
        correction = 10 ** (0.4 * data["extinction"][tile]) * 10 ** (
            -0.4 * np.median(scaled_magnitude.values, axis=0)
        )
        record_tile["flux"] = data["flux"][tile] * correction.values
        record_tile["fluxerr"] = data["fluxerr"][tile] * correction.values

        # record other important quantities
        record_tile["redshift"] = data["redshift"][tile]
        record_tile["theliname"] = data["theliname"][tile]
        record_tile["maglim"] = data["maglim"][tile]
        record_tile["extinction"] = data["extinction"][tile]

        tiles[unique_names[i]] = record_tile
        print(f"Number of objects in tile {unique_names[i]} is : {sum(tile*1)}")
        # save the tiles
        hp.pickle_save(record_tile, config.paths.tiles, unique_names[i])
    return tiles
