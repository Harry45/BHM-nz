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
    record = {}

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


def assign_binlabel(config: ConfigDict, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Given the BPZ values, we just assign each object in a tomographic redshift bin.

    Args:
        config (ConfigDict): the main configuration file with all the settings.
        dataframe (pd.DataFrame): a dataframe which consists of the BPZ redshifts (Z_B)

    Returns:
        pd.DataFrame: the bin labels for each object
    """
    nobjects = dataframe.shape[0]
    dfindex = list(dataframe.index)
    nbins = len(config.redshift.bounds)
    binlabels = [f"BIN_{i}" for i in range(nbins)]

    # record the bin label
    recordbin = []

    # this is the index of the object found in the catalogue
    recordindex = []

    for i in range(nobjects):
        for index in range(nbins):
            bound_1 = config.redshift.bounds[index][0]
            bound_2 = config.redshift.bounds[index][1]

            # some edge effect (the maximum redshift in the catalogue is 1.200195, not 1.2)
            if index == nbins - 1:
                bound_2 += 0.01
            condition_1 = dataframe["Z_B"].values[i] > bound_1
            condition_2 = dataframe["Z_B"].values[i] <= bound_2
            if condition_1 and condition_2:
                recordbin.append(binlabels[index])
                recordindex.append(dfindex[i])

    df_binlabel = pd.DataFrame(recordbin, columns=["BINLABEL"], index=recordindex)
    return df_binlabel


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
    tiles = {}
    for i in range(config.ntiles):
        record_tile = {}

        # find the objects within a particular tile
        tile = data["theliname"] == unique_names[i]
        tile = tile.values

        # calculate the correction term
        scaled_magnitude = data["mag"][tile] + 2.5 * np.log10(data["flux"][tile].values)
        correction = 10 ** (0.4 * data["extinction"][tile]) * 10 ** (
            -0.4 * np.median(scaled_magnitude.values, axis=0)
        )

        # correct for magnitude, flux and flux error
        record_tile["mag"] = data["mag"][tile] - data["extinction"][tile].values
        record_tile["flux"] = data["flux"][tile] * correction.values
        record_tile["fluxerr"] = data["fluxerr"][tile] * correction.values

        # record other important quantities
        record_tile["magerr"] = data["magerr"][tile]
        record_tile["redshift"] = data["redshift"][tile]
        record_tile["theliname"] = data["theliname"][tile]
        record_tile["maglim"] = data["maglim"][tile]
        record_tile["extinction"] = data["extinction"][tile]

        # assign bin labels
        record_tile["binlabel"] = assign_binlabel(config, record_tile["redshift"])

        # record that specific tile
        tiles[unique_names[i]] = record_tile

        # save the tiles
        hp.pickle_save(record_tile, config.paths.tiles, unique_names[i])

        print(f"Number of objects in tile {unique_names[i]} is : {sum(tile*1)}")
    return tiles


def aggregate_data(config: ConfigDict, tiles: dict) -> dict:
    """
    All the data from the different tiles are aggregated in a single dictionary with the keywords:
    redshift, binlabel, flux, fluxerr, mag, magerr

    Args:
        config (ConfigDict): the main configuration file with all the settings.
        tiles (dict): a dictionary with the data for the different tiles. This is generated from the
        function correct_data.

    Returns:
        dict: a dictionary with the different quantities and an example of how to access the quantity is:

              record['BIN_0]['redshift'].head()
    """

    nbins = len(config.redshift.bounds)
    binlabels = [f"BIN_{i}" for i in range(nbins)]

    # sb means specific bin
    sb_redshift = {binlabels[i]: [] for i in range(nbins)}
    sb_binlabel = {binlabels[i]: [] for i in range(nbins)}
    sb_flux = {binlabels[i]: [] for i in range(nbins)}
    sb_fluxerr = {binlabels[i]: [] for i in range(nbins)}
    sb_mag = {binlabels[i]: [] for i in range(nbins)}
    sb_magerr = {binlabels[i]: [] for i in range(nbins)}

    for tile in tiles:
        for binlabel in binlabels:
            condition = tiles[tile]["binlabel"] == binlabel
            condition = condition.values

            # record all the quantities we need
            sb_redshift[binlabel].append(tiles[tile]["redshift"][condition])
            sb_binlabel[binlabel].append(tiles[tile]["binlabel"][condition])
            sb_flux[binlabel].append(tiles[tile]["flux"][condition])
            sb_fluxerr[binlabel].append(tiles[tile]["fluxerr"][condition])
            sb_mag[binlabel].append(tiles[tile]["mag"][condition])
            sb_magerr[binlabel].append(tiles[tile]["magerr"][condition])
    record = {}
    record["redshift"] = sb_redshift
    record["binlabel"] = sb_binlabel
    record["flux"] = sb_flux
    record["fluxerr"] = sb_fluxerr
    record["mag"] = sb_mag
    record["magerr"] = sb_magerr
    return record


def data_per_bin(quantities: dict, binnumber: int = 0) -> dict:
    """
    Creates a dictionary which contains the different quantities for each tomographic bin.

    Args:
        quantities (dict): a dictionary containing the quantities for each bin, for example,
        quantities['redshift]['BIN_0']. This is generated from the function aggregate_data
        binnumber (int, optional): The tomographic bin number we want to look at. Defaults to 0.

    Returns:
        dict: a dictionary with all the quantities for that particular bin.
    """
    record_bins = {}
    binlabel = f"BIN_{binnumber}"
    record_bins["redshift"] = pd.concat(quantities["redshift"][binlabel])
    record_bins["flux"] = pd.concat(quantities["flux"][binlabel])
    record_bins["mag"] = pd.concat(quantities["mag"][binlabel])
    record_bins["fluxerr"] = pd.concat(quantities["fluxerr"][binlabel])
    record_bins["magerr"] = pd.concat(quantities["magerr"][binlabel])
    record_bins["binlabel"] = pd.concat(quantities["binlabel"][binlabel])
    return record_bins


def data_all_bins(config: ConfigDict, agg_data: dict) -> dict:
    """
    Gather the data from each bin in the following format:

    record['BIN_0]['redshift']

    and so forth

    Args:
        config (ConfigDict): the main configuration file with all the settings.
        agg_data (dict): the aggregated data, generated using the function data_per_bin.

    Returns:
        dict: a dictionary with the quantities for all the bins.
    """
    record_bins = {}
    nbins = len(config.redshift.bounds)
    for i in range(nbins):
        record_bins[f"BIN_{i}"] = data_per_bin(agg_data, binnumber=i)
    return record_bins
