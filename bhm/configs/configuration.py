"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: The main configuration file for the project.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""
import ml_collections


def get_config():

    config = ml_collections.ConfigDict()

    # bands
    config.band = ['u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'Ks']

    # flux
    config.flux = flux = ml_collections.ConfigDict()
    config.flux.names = [f'FLUX_GAAP_{b}' for b in config.band]
    config.flux.err_names = [f'FLUXERR_GAAP_{b}' for b in config.band]

    return config
