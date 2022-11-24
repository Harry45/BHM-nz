"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: The main configuration file for the project.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""

from ml_collections.config_dict import ConfigDict


def get_config() -> ConfigDict:
    """The main configuration file for processing and generating the n(z) for KiDS-1000.

    Returns:
        ConfigDict: all configurations.
    """

    config = ConfigDict()

    # bands
    config.band = ['u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'Ks']

    # flux
    config.flux = flux = ConfigDict()
    flux.value_cols = [f'FLUX_GAAP_{b}' for b in config.band]
    flux.err_cols = [f'FLUXERR_GAAP_{b}' for b in config.band]

    # magnitudes
    config.mag = mag = ConfigDict()
    mag.value_cols = [f'MAG_GAAP_{b}' for b in config.band]
    mag.err_cols = [f'MAGERR_GAAP_{b}' for b in config.band]

    # columns related to BPZ
    config.bpz_cols = ['Z_B', 'M_0', 'Z_ML', 'Z_B_MIN', 'Z_B_MAX']

    # other important columns
    config.meta_cols = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'weight']

    # some quantities for size of arrays
    config.nbpz = len(config.bpz_cols)
    config.nmeta = len(config.meta_cols)
    config.nband = len(config.band)

    # redshift
    config.redshift = redshift = ConfigDict()
    redshift.bounds = [(0.1, 0.3),
                       (0.3, 0.5),
                       (0.5, 0.7),
                       (0.7, 0.9),
                       (0.9, 1.2)]
    redshift.range = [0.0, 2.0]

    return config
