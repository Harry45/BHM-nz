import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
import pandas as pd
from getdist import plots, MCSamples
from collections import Counter


def prob_template_given_mag(fraction: float, kt: float, magnitude: float) -> float:
    """
    Calculates the probability of the template given the magnitude. See equation in Benitez (2000).

    | Spectral type | Fraction | kt    |
    |---------------|----------|-------|
    | E/SO          | 0.35     | 0.147 |
    | Spirals       | 0.50     | 0.450 |
    | Irregulars    | 0.15     |       |

    Args:
        fraction (float): the fraction of each type of object.
        kt (float): a parameter in the equation.
        magnitude (float): the magnitude value

    Returns:
        float: the population of each type.
    """

    if magnitude > 32.0:
        magnitude = 32.0 + 1e-32
    if magnitude < 20.0:
        magnitude = 20.0 + 1e-32

    expterm = kt * (magnitude - 20.0)
    return fraction * np.exp(-expterm)


def prob_redshift_given_template_mag(
    redshift: np.ndarray,
    magnitude: float,
    alpha: float,
    redshift_ref: float,
    kmt: float,
) -> np.ndarray:
    """
    Calculates the pdf of redshift given magnitude and template. See Benitez (2000).

    | Spectral type | alpha  | redshift reference | kmt    |
    |---------------|--------|--------------------|--------|
    | E/SO          | 2.465  | 0.431              | 0.0913 |
    | Spirals       | 1.806  | 0.390              | 0.0636 |
    | Irregulars    | 0.906  | 0.0626             | 0.123  |

    Args:
        redshift (np.ndarray): the redshift on a grid within two limits, for example, 0 and 3.
        magnitude (float): the magnitude of the object.
        alpha (float): a parameter in the equation (depends on the object type).
        redshift_ref (float): another parameter in the equation.
        kmt (float): another parameter in the equation.

    Returns:
        np.ndarray: the normalised pdf of redshift given magnitude and type.
    """
    if magnitude > 32.0:
        magnitude = 32.0 + 1e-32
    if magnitude < 20.0:
        magnitude = 20.0 + 1e-32

    zmt = redshift_ref + kmt * (magnitude - 20)
    probability = redshift**alpha * np.exp(-((redshift / zmt) ** alpha))
    norm = np.trapz(probability, redshift)
    probability = probability / norm
    return probability


def prob_redshift_given_mag(redshift: np.ndarray, magnitude: float) -> np.ndarray:
    """
    Calculates the probability of redshift given magnitude. This is marginalised over templates.

    Args:
        redshift (np.ndarray): the redshift on a grid.
        magnitude (float): the magnitude value.

    Returns:
        np.ndarray: the normalised probability distribution.
    """
    p_temp_eso = prob_template_given_mag(0.35, 0.147, magnitude)
    p_temp_spi = prob_template_given_mag(0.50, 0.45, magnitude)
    p_temp_irr = 1 - (p_temp_eso + p_temp_spi)

    probability_eso = prob_redshift_given_template_mag(
        redshift, magnitude, 2.465, 0.431, 0.0913
    )
    probability_spi = prob_redshift_given_template_mag(
        redshift, magnitude, 1.806, 0.390, 0.0636
    )
    probability_irr = prob_redshift_given_template_mag(
        redshift, magnitude, 0.906, 0.0626, 0.123
    )

    probability = (
        p_temp_eso * probability_eso
        + p_temp_spi * probability_spi
        + p_temp_irr * probability_irr
    )
    probability = probability / np.trapz(probability, redshift)
    return probability


def interpolate_filters(folder: str, nwave: int = 1000) -> dict:
    """
    Interpolate the filters on a common wavelength. We first record the minimum and
    maximum of each filter and we find the minimum of minimum and maximum of maximum.
    We then output a dictionary which contains the filters and the interpolated wavelengths.

    Args:
        folder (str): name of the folder where the filters are stored.
        nwave (int, optional): number of wavelength between the minimum and maximum. Defaults to 1000.

    Returns:
        dict: a dictionary with the filters and the intepolated wavelength.
    """
    fnames = os.listdir(folder)
    filters = dict()
    filters_int = dict()
    record_min = list()
    record_max = list()
    for file in fnames:
        filterband = np.loadtxt(f"{folder}/{file}")[:, [0, 1]]
        wavelength = filterband[:, 0]
        record_min.append(min(wavelength))
        record_max.append(max(wavelength))
        filters[file.split(".")[0]] = filterband

    minwave = min(record_min)
    maxwave = max(record_max)
    waveint = np.linspace(minwave, maxwave, nwave)

    for key, value in filters.items():
        func = interp1d(
            value[:, 0],
            value[:, 1],
            bounds_error=False,
            kind="linear",
            fill_value=(value[:, 1][0], value[:, 1][-1]),
        )
        ynew = func(waveint)
        filters_int[key] = ynew
    filters_int["wavelength"] = waveint
    return filters_int


def process_templates(
    folder: str, waverange: list, nwave: int = 1000, normwave=7e3
) -> dict:
    """
    Process the templates/seds in such a way that they have a common wavelength.

    Args:
        folder (str): the folder where the templates/seds are stored.
        waverange (list): the minimum and maximum of the wavelength.
        nwave (int, optional): the number of wavelengths to use in between. Defaults to 1000.
        normwave (float, optional): the wavelength at which the templates are normalised. Defaults to 7E3.

    Returns:
        dict: a dictionary with the templates and the interpolated wavelengths.
    """
    waveint = np.linspace(waverange[0], waverange[1], nwave)
    templates = dict()
    seds = os.listdir(folder)
    for f in seds:
        file = np.loadtxt(os.path.join(folder, f))
        wavelength = file[:, 0]
        sed = file[:, 1] * file[:, 0] ** 2 / 3e18
        norm_constant = np.interp(normwave, wavelength, sed)
        sed = sed / norm_constant
        func = interp1d(
            wavelength,
            sed,
            bounds_error=False,
            kind="linear",
            fill_value=(sed[0], sed[-1]),
        )
        ynew = func(waveint)
        templates[f.split(".")[0]] = ynew
    templates["wavelength"] = waveint
    return templates


def imagnitude_distribution(
    mgrid: np.ndarray,
    alpha: float = 15.0,
    beta: float = 2.0,
    maglim: float = 24.0,
    offset: float = 1.0,
) -> np.ndarray:
    """
    Generates the distribution for the magnitude (using i here).

    Args:
        mgrid (np.ndarray): the grid of magnitude to consider.
        alpha (float, optional): parameter in the model. Defaults to 15.0.
        beta (float, optional): parameter in the model. Defaults to 2.0.
        maglim (float, optional): magnitude limit. Defaults to 24.0.
        offset (float, optional): an offset parameter. Defaults to 1.0.

    Returns:
        np.ndarray: the normalised probability distribution of the magnitude
    """
    probability = mgrid**alpha * np.exp(-((mgrid / (maglim - offset)) ** beta))
    normalisation = np.trapz(probability, mgrid)
    probability /= normalisation
    return probability


def imagnitude_error_distribution(
    magnitudes: np.ndarray,
    param_a: float = 4.56,
    param_b: float = 1.0,
    param_k: float = 1.0,
    maglim: float = 24.0,
    sigmadet: float = 5.0,
) -> np.ndarray:
    """
    Error distribution for the magnitudes. Following Rykoff et al. 2015.

    Args:
        magnitudes (np.ndarray): the magnitudes.
        param_a (float, optional): a parameter in the model. Defaults to 4.56.
        param_b (float, optional): a parameter in the model. Defaults to 1.0.
        param_k (float, optional): a parameter in the model. Defaults to 1.0.
        maglim (float, optional): the magnitude limit. Defaults to 24.0.
        sigmadet (float, optional): level of sigma detection. Defaults to 5.0.

    Returns:
        np.ndarray: the magnitude error.
    """
    teff = np.exp(param_a + param_b * (maglim - 21.0))
    flux = 10 ** (-0.4 * (magnitudes - 22.5))
    flim = 10 ** (-0.4 * (maglim - 22.5))
    fnoise = (flim / sigmadet) ** 2 * param_k * teff - flim
    sigma_m = (
        2.5 / np.log(10) * np.sqrt((1.0 + fnoise / flux) / (flux * param_k * teff))
    )
    return sigma_m


def draw_samples_one_dim(
    pdf: np.ndarray, grid: np.ndarray, nsamples: int
) -> np.ndarray:
    """
    Draw samples from a one-dimensional pdf.

    Args:
        pdf (np.ndarray): the probability density function.
        grid (np.ndarray): the support.
        nsamples (int): the number of samples we want

    Returns:
        np.ndarray: the samples.
    """
    cdf = np.cumsum(pdf)
    cdf /= np.max(cdf)
    uniform = np.random.uniform(0, 1, nsamples)
    func = interp1d(cdf, grid, bounds_error=False)
    samples = func(uniform)
    return samples


def analytical_error_distribution(
    probability_imag: np.ndarray, mgrid: np.ndarray, maglim: float = 24.0
) -> np.ndarray:
    """
    Get an analytical error distribution for the magnitude given the pdf of the magnitude.

    Args:
        probability_imag (np.ndarray): the probability distribution (pdf) of the magnitude.
        mgrid (np.ndarray): the support for the magnitude.
        maglim (float, optional): the magnitude limit. Defaults to 24.0.

    Returns:
        np.ndarray: the new pdf error magnitude based on the magnitude limit.
    """
    detprob = np.copy(probability_imag)
    ind = mgrid >= maglim - 0.4
    detprob[ind] *= np.exp(-0.5 * ((mgrid[ind] - maglim + 0.4) / 0.2) ** 2)
    normalisation = np.trapz(detprob, mgrid)
    detprob /= normalisation
    return detprob


def pdf_redshift(
    magnitude: float, redshift: np.ndarray, template_ref: str = "eso"
) -> np.ndarray:
    """
    Generates the redshift distribution based on magnitude and the reference template.

    Args:
        magnitude (float): the magnitude value.
        redshift (np.ndarray): the grid of redshift.
        template_ref (str, optional): the reeference template. Can also set to 'marg' if we want the
        marginalised redshift distribution. Defaults to 'eso'.

    Returns:
        np.ndarray: the probability density function.
    """

    assert template_ref in [
        "eso",
        "spi",
        "irr",
        "marg",
    ], "template reference should be one of eso, spi, irr, marg"

    if template_ref == "eso":
        pdf = prob_redshift_given_template_mag(
            redshift, magnitude, 2.465, 0.431, 0.0913
        )
    elif template_ref == "spi":
        pdf = prob_redshift_given_template_mag(
            redshift, magnitude, 1.806, 0.390, 0.0636
        )
    elif template_ref == "irr":
        pdf = prob_redshift_given_template_mag(
            redshift, magnitude, 0.906, 0.0626, 0.123
        )
    else:
        pdf = prob_redshift_given_mag(redshift, magnitude)
    return pdf


def sample_redshift(
    magnitude: float, redshift: np.ndarray, template_ref: str = "eso"
) -> np.ndarray:
    """
    Generate a sample of redshift given the support and the reference template.

    Args:
        magnitude (float): the value of magnitude.
        redshift (np.ndarray): the support for the redshift.
        template_ref (str, optional): the template reference. Defaults to "eso".

    Returns:
        np.ndarray: the redshift sample.
    """
    pdf = pdf_redshift(magnitude, redshift, template_ref)
    sample = draw_samples_one_dim(pdf, redshift, 1)
    return sample


def generate_2d_plot(samples: np.ndarray, name: str = "E/SO") -> plots.GetDistPlotter:
    """
    Generates a 2D density plot of the samples of redshift and magnitudes. The first column contains the redshift
    while the second column contains the magnitudes.

    Args:
        samples (np.ndarray): the redshift and magnitude samples.
        name (str, optional): the name of the template type. Defaults to "E/SO".

    Returns:
        getdist.plots.GetDistPlotter: the getdist plot.
    """
    ndim = samples.shape[1]
    names = ["x%s" % i for i in range(ndim)]
    labels = [r"$z$", r"$m_{i}$"]
    samps = MCSamples(
        samples=samples,
        names=names,
        labels=labels,
        ranges={"x0": (0.0, None)},
        settings=settings,
    )
    samps.updateSettings({"contours": [0.68, 0.95, 0.99]})

    ax = plots.get_single_plotter(width_inch=4, ratio=1)
    ax.settings.axes_fontsize = 15
    ax.settings.lab_fontsize = 15
    ax.settings.num_plot_contours = 3
    ax.settings.solid_contour_palefactor = 0.75
    ax.settings.alpha_filled_add = 0.9
    ax.plot_2d(samps, "x0", "x1", filled=True, colors=["red"], lims=[0.0, 4.0, 16, 26])
    ax.add_legend([name], colored_text=True, legend_loc="lower right")
    return ax


def plot_all_samples(
    data_eso: np.ndarray,
    data_spi: np.ndarray,
    data_irr: np.ndarray,
    data_marg: np.ndarray,
) -> None:
    """
    Plot the 2D distribution for the samples obtained from different templates, as well as,
    the one where we marginalise over all the three templates. The first column contains the
    redshift samples and the second column contains the magnitude samples.

    Args:
        data_eso (np.ndarray): the redshift and magnitude samples using the E/SO type template.
        data_spi (np.ndarray): the redshift and magnitude samples using the Spiral type template.
        data_irr (np.ndarray): the redshift and magnitude samples using the Irregular type template.
        data_marg (np.ndarray): the redshift and magnitude samples when the templates are marginalised over.
    """
    newsamples = np.concatenate([data_eso, data_spi, data_irr, data_marg], axis=1)

    limits = {
        "x0": (0.0, None),
        "x1": (16, 26),
        "x2": (0.0, None),
        "x3": (16, 26),
        "x4": (0.0, None),
        "x5": (16, 26),
        "x6": (0.0, None),
        "x7": (16, 26),
    }

    ndim = 8
    names = ["x%s" % i for i in range(ndim)]
    labels = [r"$z$", r"$m_{i}$"] * 4
    samps = MCSamples(
        samples=newsamples, names=names, labels=labels, ranges=limits, settings=settings
    )

    g = plots.get_subplot_plotter(subplot_size=4)
    g.settings.axes_fontsize = 15
    g.settings.lab_fontsize = 15
    g.settings.num_plot_contours = 3
    g.settings.solid_contour_palefactor = 0.75
    g.settings.alpha_filled_add = 0.9
    g.settings.scaling = False
    g.plots_2d(
        samps,
        param_pairs=[["x0", "x1"], ["x2", "x3"], ["x4", "x5"], ["x6", "x7"]],
        nx=2,
        filled=True,
        colors=["red"],
        lims=[0.0, 4.0, 16, 26],
    )
    plt.show()


def plot_1d_redshift_distributions(
    redshift_grid: np.ndarray, pdfs: list, nselected: int
) -> None:
    """
    Plot samples of the redshift distribution.

    Args:
        redshift_grid (np.ndarray): the redshift support.
        pdfs (list): a list of the redshift distributions
        nselected (int): the number of selected redshift distribution.
    """

    assert (
        nselected < pdfs[0].shape[0]
    ), f"Number of samples to be plotted should be less than {pdfs[0].shape[0]}."

    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.title("E/SO", fontsize=fontSize)
    plt.plot(redshift_grid, pdfs[0][0:nselected].T, lw=0.1)
    plt.xlabel(r"$z$", fontsize=fontSize)
    plt.ylabel(r"$p(z|T_{1}, m)$", fontsize=fontSize)
    plt.tick_params(axis="x", labelsize=fontSize)
    plt.tick_params(axis="y", labelsize=fontSize)
    plt.xlim(0, 3.0)
    plt.ylim(0, 2.5)

    plt.subplot(222)
    plt.title("Spirals", fontsize=fontSize)
    plt.plot(redshift_grid, pdfs[1][0:nselected].T, lw=0.1)
    plt.xlabel(r"$z$", fontsize=fontSize)
    plt.ylabel(r"$p(z|T_{2}, m)$", fontsize=fontSize)
    plt.tick_params(axis="x", labelsize=fontSize)
    plt.tick_params(axis="y", labelsize=fontSize)
    plt.xlim(0, 3.0)
    plt.ylim(0, 2.0)

    plt.subplot(223)
    plt.title("Irregulars", fontsize=fontSize)
    plt.plot(redshift_grid, pdfs[2][0:nselected].T, lw=0.1)
    plt.xlabel(r"$z$", fontsize=fontSize)
    plt.ylabel(r"$p(z|T_{3}, m)$", fontsize=fontSize)
    plt.tick_params(axis="x", labelsize=fontSize)
    plt.tick_params(axis="y", labelsize=fontSize)
    plt.xlim(0, 3.0)
    plt.ylim(0, 6.0)

    plt.subplot(224)
    plt.title("Marginalised over templates", fontsize=fontSize)
    plt.plot(redshift_grid, pdfs[3][0:nselected].T, lw=0.1)
    plt.xlabel(r"$z$", fontsize=fontSize)
    plt.ylabel(r"$p(z|m)$", fontsize=fontSize)
    plt.tick_params(axis="x", labelsize=fontSize)
    plt.tick_params(axis="y", labelsize=fontSize)
    plt.xlim(0, 3.0)
    plt.ylim(0, 2.0)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.45
    )
    plt.show()


def approx_lum_dist(redshift: float) -> float:
    """
    Calculates the approximate luminosity distance given a redshift.

    Args:
        redshift (float): the redshift value.

    Returns:
        float: the approximate luminosity distance.
    """
    return np.exp(30.5 * redshift**0.04 - 21.7)


def get_seds(folder: str, waveref: float = 4e5) -> dict:
    """
    Get the SEDs from the folder and the SEDs are normalised at a particular wavelength.

    Args:
        folder (str): folder where the SEDs are stored.
        waveref (float, optional): the wavelength at which the SEDs are standardised. Defaults to 4e5.

    Returns:
        dict: dictionary with the standardised SEDs.
    """
    sednames = os.listdir(folder)
    seds = dict()
    for i, t in enumerate(sednames):
        seddata = np.genfromtxt(f"{folder}/{t}")
        seddata[:, 1] *= seddata[:, 0] ** 2.0 / 3e18
        ref = np.interp(waveref, seddata[:, 0], seddata[:, 1])
        seddata[:, 1] /= ref
        seds[t.split(".")[0]] = seddata
    return seds


def get_filters(folder: str) -> Tuple[dict, dict, dict]:
    """
    Get the filters to be used to calculate the flux.

    Args:
        folder (str): folder where the filters are stored.

    Returns:
        Tuple[dict, dict, dict]: the filter, its wavelength range, the minimum and maximum wavelength.
    """
    filters = os.listdir(folder)
    record = dict()
    record_wavelength = dict()
    record_minmax = dict()

    for f in filters:
        data = np.genfromtxt(os.path.join(folder, f))
        wavelength, filt = data[:, 0], data[:, 1]
        filt /= wavelength
        norm = np.trapz(filt, wavelength)
        filt /= norm

        # find minimum and maximum wavelength
        ind = np.where(filt > 0.01 * np.max(filt))[0]
        lambda_min, lambda_max = wavelength[ind[0]], wavelength[ind[-1]]

        record[f.split(".")[0]] = filt
        record_minmax[f.split(".")[0]] = [lambda_min, lambda_max]
        record_wavelength[f.split(".")[0]] = wavelength

    return record, record_wavelength, record_minmax


def get_flux(
    filters: dict, wavelengths: dict, lambdaminmax: dict, seds: dict, redshift: float
) -> pd.DataFrame:
    """
    Calculate the theoretical flux. The output is a dataframe of size Nt x Nf,
    where Nt is the number of templates and Nf is the number of filters.

    Args:
        filters (dict): a dictionary of filters.
        wavelengths (dict): a dictionary of the wavelength range.
        lambdaminmax (dict): the minimum and maximum of wavelength.
        seds (dict): a dictionary with all the SEDs/templates.
        redshift (float): the redshift value.

    Returns:
        pd.DataFrame: a dataframe of fluxes for Nt templates and Nf filters.
    """
    scaled_redshift = 1.0 + redshift
    nfilters = len(filters)
    nseds = len(seds)

    # calculate pre-factor
    lum_dist = approx_lum_dist(redshift)
    prefactor = scaled_redshift**2 / (4.0 * np.pi * lum_dist**2)

    # empty array to store the fluxes
    record_fluxes = dict()

    for i, s in enumerate(seds):
        fluxes = np.zeros(nfilters)
        for j, f in enumerate(filters):
            wave_grid = np.linspace(
                lambdaminmax[f][0] / scaled_redshift,
                lambdaminmax[f][1] / scaled_redshift,
                5000,
            )
            filter_interp = interp1d(wavelengths[f] / scaled_redshift, filters[f])
            sed_interp = interp1d(seds[s][:, 0], seds[s][:, 1])
            filter_new = filter_interp(wave_grid)
            sed_new = sed_interp(wave_grid)
            fluxes[j] = prefactor * np.trapz(sed_new * filter_new, wave_grid)
        record_fluxes[s] = fluxes
    record_fluxes = pd.DataFrame(record_fluxes).T
    record_fluxes.columns = filters.keys()
    return record_fluxes


def sample_mag_sigma(p_imag: np.ndarray, imag_grid: np.ndarray) -> Tuple[float, float]:
    """
    Sample a value of magnitude and its corresponding error. We do this for high signal to noise ratio,
    that is, we set the threshold sigma level to 5.

    Args:
        p_imag (np.ndarray): the pdf of the magnitude.
        imag_grid (np.ndarray): the support for the magnitude.

    Returns:
        Tuple[float, float]: the magnitude sample and a sample of error.
    """
    sigmalevel = 1e-32
    while sigmalevel < 5:
        mag = draw_samples_one_dim(p_imag, imag_grid, 1)
        err = imagnitude_error_distribution(mag)
        sigmalevel = 1 / err
    return mag, err


def sample_template(magnitude: float, template_match: dict) -> Tuple[str, str]:
    """
    Sample a template based on probabilities.

    Args:
        magnitude (float): the magnitude value.
        template_match (dict): a dictionary which describes the nature of the template, for example,

    template_match = {
    'El_B2004a': 'eso',
    'Sbc_B2004a': 'spi',
    'Scd_B2004a': 'spi',
    'Im_B2004a': 'irr',
    'SB3_B2004a': 'irr',
    'SB2_B2004a': 'irr',
    'ssp_25Myr_z008': 'irr',
    'ssp_5Myr_z008': 'irr'
    }

    Returns:
        Tuple[str, str]: the template name and the template type.
    """
    counts = dict(Counter([value[1] for value in template_match.items()]))
    p_temp_eso = prob_template_given_mag(0.35, 0.147, magnitude)
    p_temp_spi = prob_template_given_mag(0.50, 0.45, magnitude)
    p_temp_irr = 1 - (p_temp_eso + p_temp_spi)
    probabilities = {
        "eso": p_temp_eso / counts["eso"],
        "spi": p_temp_spi / counts["spi"],
        "irr": p_temp_irr / counts["irr"],
    }

    probs = dict()
    for key in template_match.keys():
        if template_match[key] == "eso":
            probs[key] = probabilities["eso"]
        elif template_match[key] == "spi":
            probs[key] = probabilities["spi"]
        else:
            probs[key] = probabilities["irr"]

    probabilities = np.asarray(list(probs.values())).reshape(-1)
    temp_samp = np.random.choice(list(probs.keys()), p=probabilities)
    return temp_samp, template_match[temp_samp]
