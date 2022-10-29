# BHM-nz

Bayesian Hierarchical Modelling of Redshift Distributions

[BL-2016](https://github.com/Harry45/BHM-nz/tree/main/reference/BL-2016) contains the
original implementation of the Bayesian Hierarchical Model by Boris and
[KiDS-1000](https://github.com/Harry45/BHM-nz/tree/main/reference/KiDS-1000) contains the
original KiDS-1000 likelihood function.

The main goal of this project is to infer the tomographic redshift distribution
of the data (survey) using the Bayesian Hierarchical Model methodology. These
distributions can then be used to compute the weak lensing power spectra, and
then infer cosmological parameters, whilst marginalising over the nuisance
parameters. In this way, one does not have to shift the distribution and hence
these shifts are set to zero. The marginalisation is done by drawing random
samples of the heights at each step in the likelihood calculation.
