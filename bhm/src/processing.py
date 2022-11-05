"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: Function to process the data in the right format.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""
import os
import sys
import random
import pandas as pd
import numpy as np
from astropy.io import fits

from configuration import config
import utils.helpers as hp

CONFIG = config()


def get_size_mb(dataframe: pd.DataFrame, name: str):
    """Print the size of the pandas dataframe in MB

    Args:
        dataframe (pd.DataFrame): the pandas dataframe
        name (str): name of the dataframe
    """
    print(f'Size of {name} dataframe {sys.getsizeof(dataframe)/1024**2:.2f} MB')


def cleaning(catalogue: fits, save: bool, **kwargs) -> dict:

    # the data from the catalogue
    data = catalogue[1].data

    if 'ngalaxies' in kwargs:
        ngalaxies = kwargs.pop('ngalaxies')
        indices = random.sample(len(data), ngalaxies)
        data = data[indices]

    # important columns in the catalogue
    fluxes = np.asarray([data[CONFIG.flux.value_cols[i]] for i in range(CONFIG.nband)]).T
    fluxes_err = np.asarray([data[CONFIG.flux.err_cols[i]] for i in range(CONFIG.nband)]).T
    mag = np.asarray([data[CONFIG.mag.value_cols[i]] for i in range(CONFIG.nband)]).T
    mag_err = np.asarray([data[CONFIG.mag.err_cols[i]] for i in range(CONFIG.nband)]).T
    bpz = np.asarray([data[CONFIG.bpz_cols[i]] for i in range(CONFIG.nbpz)]).T
    meta = np.asarray([data[CONFIG.meta_cols[i]] for i in range(CONFIG.nmeta)]).T

    # Convert everything to pandas dataframes
    df_flux = pd.DataFrame(fluxes, columns=CONFIG.flux.value_cols, dtype=np.float32)
    df_flux_err = pd.DataFrame(fluxes_err, columns=CONFIG.flux.err_cols, dtype=np.float32)
    df_mag = pd.DataFrame(mag, columns=CONFIG.mag.value_cols, dtype=np.float16)
    df_mag_err = pd.DataFrame(mag_err, columns=CONFIG.mag.err_cols, dtype=np.float16)
    df_bpz = pd.DataFrame(bpz, columns=CONFIG.bpz_cols, dtype=np.float16)
    df_meta = pd.DataFrame(meta, columns=CONFIG.meta_cols, dtype=np.float16)

    # Print the size of dataframes
    get_size_mb(df_flux, 'flux')
    get_size_mb(df_flux_err, 'flux errors')
    get_size_mb(df_mag, 'magnitude')
    get_size_mb(df_mag_err, 'magnitude errors')
    get_size_mb(df_bpz, 'BPZ')
    get_size_mb(df_meta, 'meta')

    dictionary = {}
    dictionary['flux'] = df_flux
    dictionary['flux_err'] = df_flux_err
    dictionary['mag'] = df_mag
    dictionary['mag_err'] = df_mag_err
    dictionary['bpz'] = df_bpz
    dictionary['meta'] = df_meta

    if save:
        folder = kwargs.pop('folder')
        os.makedirs(folder, exist_ok=True)

        # save the files
        hp.pickle_save(df_flux, folder, 'flux')
        hp.pickle_save(df_flux_err, folder, 'flux_err')
        hp.pickle_save(df_mag, folder, 'mag')
        hp.pickle_save(df_mag_err, 'mag_err')
        hp.pickle_save(df_bpz, 'bpz')
        hp.pickle_save(df_meta, 'meta')
    return dictionary
