"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: The main configuration file for the project.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""
import numpy as np
from ml_collections.config_dict import ConfigDict


def get_config(experiment: str) -> ConfigDict:
    """The main configuration file for processing and generating the n(z) for KiDS-1000.
    Args:
        experiment (str): name of the experiment

    Returns:
        ConfigDict: all configurations.
    """

    config = ConfigDict()
    config.experiment = experiment

    # some default values
    config.ntiles = 5

    # paths
    config.paths = paths = ConfigDict()
    paths.fitsfile = "data/catalogue/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits"
    paths.tiles = "data/tiles/"

    # bands
    config.band = ["u", "g", "r", "i", "Z", "Y", "J", "H", "Ks"]
    config.nband = len(config.band)

    # column names
    config.colnames = colnames = ConfigDict()
    colnames.flux = [f"FLUX_GAAP_{b}" for b in config.band]
    colnames.fluxerr = [f"FLUXERR_GAAP_{b}" for b in config.band]
    colnames.mag = [f"MAG_GAAP_{b}" for b in config.band]
    colnames.magerr = [f"MAGERR_GAAP_{b}" for b in config.band]
    colnames.extinction = [f"EXTINCTION_{b}" for b in config.band]
    colnames.maglim = [f"MAG_LIM_{b}" for b in config.band]
    colnames.theliname = ["THELI_NAME"]
    colnames.redshift = ["Z_B", "Z_ML"]

    # dtypes
    config.dtypes = dtypes = ConfigDict()
    dtypes.flux = np.float32
    dtypes.fluxerr = np.float32
    dtypes.mag = np.float16
    dtypes.magerr = np.float16
    dtypes.extinction = np.float16
    dtypes.maglim = np.float16
    dtypes.theliname = str
    dtypes.redshift = np.float16

    # redshift
    config.redshift = redshift = ConfigDict()
    redshift.bounds = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.2)]
    redshift.range = [0.0, 2.0]

    return config
