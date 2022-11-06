"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: Function to stack the redshift and kind the kernel density estimate.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""
import numpy as np
from sklearn.neighbors import KernelDensity

from configuration import config
import utils.helpers as hp

CONFIG = config()


def kde_fitting(samples: np.ndarray, bandwidth: float) -> KernelDensity:
    """Performs a Kernel Density Estimate on the samples of redshifts.

    Args:
        samples (np.ndarray): the samples of the histogram
        bandwidth (float): the bandwidth of the normal distribution

    Returns:
        KernelDensity: the fitted kernel density
    """

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(samples)

    # to add cross validation step here
    return kde


def stack_fitting(redshifts: np.ndarray, save: bool, bandwidth: float = 0.02, ngalaxies: int = None) -> dict:
    """Takes in the redshifts from the catalogue and split it into tomographic bins. Also performs a Kernel Density Estimate
    of the samples within each bin and store them in the stacking/ folder.

    Args:
        redshifts (np.ndarray): the redshifts from the catalogue.
        save (bool): option to save the KDEs.
        bandwidth (float, optional): the bandwidth to use for the KDE. Defaults to 0.02.
        ngalaxies (int): the number of galaxies to use if we want a fast KDE fit.

    Returns:
        dict: the fitted KDEs for each tomographic bin.
    """

    if ngalaxies is not None:
        redshifts = redshifts[0: ngalaxies]

    tomo_1 = redshifts[(redshifts > CONFIG.redshift.bounds[0][0]) & (redshifts <= CONFIG.redshift.bounds[0][1])]
    tomo_2 = redshifts[(redshifts > CONFIG.redshift.bounds[1][0]) & (redshifts <= CONFIG.redshift.bounds[1][1])]
    tomo_3 = redshifts[(redshifts > CONFIG.redshift.bounds[2][0]) & (redshifts <= CONFIG.redshift.bounds[2][1])]
    tomo_4 = redshifts[(redshifts > CONFIG.redshift.bounds[3][0]) & (redshifts <= CONFIG.redshift.bounds[3][1])]
    tomo_5 = redshifts[(redshifts > CONFIG.redshift.bounds[4][0]) & (redshifts <= CONFIG.redshift.bounds[4][1])]

    kde_1 = kde_fitting(tomo_1[:, None], bandwidth)
    kde_2 = kde_fitting(tomo_2[:, None], bandwidth)
    kde_3 = kde_fitting(tomo_3[:, None], bandwidth)
    kde_4 = kde_fitting(tomo_4[:, None], bandwidth)
    kde_5 = kde_fitting(tomo_5[:, None], bandwidth)

    if save:
        hp.pickle_save(kde_1, 'stacking', f'kde_1_{bandwidth}')
        hp.pickle_save(kde_2, 'stacking', f'kde_2_{bandwidth}')
        hp.pickle_save(kde_3, 'stacking', f'kde_3_{bandwidth}')
        hp.pickle_save(kde_4, 'stacking', f'kde_4_{bandwidth}')
        hp.pickle_save(kde_5, 'stacking', f'kde_5_{bandwidth}')

    dictionary = {'kde_1': kde_1, 'kde_2': kde_2, 'kde_3': kde_3, 'kde_4': kde_4, 'kde_5': kde_5}
    return dictionary


def stack_predictions(redshifts: np.ndarray, bandwidth: float, save: bool, fname: str) -> dict:
    """Calculates the n(z) via the stacking method for a given set of redshifts (for example the KiDS-1000 mid-redshifts)

    Args:
        redshifts (np.ndarray): the redshifts.
        bandwidth (float): the bandwidth of the KDEs used in the fitting methodology
        save (bool): save the outputs when predictions are made
        fname (str): name of the file

    Returns:
        dict: the redshifts and the normalised heights.
    """

    # load the kde pickle files
    kde_1 = hp.pickle_load('stacking', f'kde_1_{bandwidth}')
    kde_2 = hp.pickle_load('stacking', f'kde_2_{bandwidth}')
    kde_3 = hp.pickle_load('stacking', f'kde_3_{bandwidth}')
    kde_4 = hp.pickle_load('stacking', f'kde_4_{bandwidth}')
    kde_5 = hp.pickle_load('stacking', f'kde_5_{bandwidth}')

    # the heights
    height_1 = np.exp(kde_1.score_samples(redshifts[:, None]))
    height_2 = np.exp(kde_2.score_samples(redshifts[:, None]))
    height_3 = np.exp(kde_3.score_samples(redshifts[:, None]))
    height_4 = np.exp(kde_4.score_samples(redshifts[:, None]))
    height_5 = np.exp(kde_5.score_samples(redshifts[:, None]))

    # normalise the heights such that area is one
    height_1 = height_1 / np.trapz(height_1, redshifts)
    height_2 = height_2 / np.trapz(height_2, redshifts)
    height_3 = height_3 / np.trapz(height_3, redshifts)
    height_4 = height_4 / np.trapz(height_4, redshifts)
    height_5 = height_5 / np.trapz(height_5, redshifts)

    dictionary = {'redshifts': redshifts, 'h1': height_1,
                  'h2': height_2, 'h3': height_3, 'h4': height_4, 'h5': height_5}
    if save:
        hp.pickle_save(dictionary, 'stacking', fname)
    return dictionary
